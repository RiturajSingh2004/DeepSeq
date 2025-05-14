from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body, Depends
from typing import Dict, List, Optional, Union, Any
import logging
import numpy as np

from app.api.models.requests import (
    PropertyPredictionRequest,
    BatchPropertyRequest,
    UniProtRequest
)
from app.api.models.responses import (
    PropertyPredictionResponse,
    BatchPropertyResponse,
    ErrorResponse
)
from app.core.property_prediction import get_property_predictor, AVAILABLE_PROPERTIES
from app.services.sequence_utils import validate_sequence, process_sequence, get_sequence_from_uniprot
from app.services.file_handling import get_file_handler

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict", response_model=PropertyPredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_properties(request: PropertyPredictionRequest):
    """
    Predict properties for a protein sequence.
    
    Parameters:
        request: Property prediction request with sequence and properties to predict
    
    Returns:
        Predicted property values for each residue
    """
    try:
        # Process and validate sequence
        sequence = process_sequence(request.sequence)
        
        if not validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Validate requested properties
        for prop in request.properties:
            if prop not in AVAILABLE_PROPERTIES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unknown property: {prop}. Available properties: {AVAILABLE_PROPERTIES}"
                )
        
        # Get property predictor
        predictor = get_property_predictor()
        
        # Generate predictions
        results = []
        for prop_name in request.properties:
            try:
                prediction = predictor.predict_property(sequence, prop_name)
                
                # Convert numpy arrays to lists if needed
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.tolist()
                
                results.append({
                    "property_name": prop_name,
                    "values": prediction,
                    "metadata": {
                        "min": float(np.min(prediction)),
                        "max": float(np.max(prediction)),
                        "mean": float(np.mean(prediction))
                    }
                })
            except Exception as e:
                logger.error(f"Error predicting {prop_name}: {e}")
                # Add empty result for failed prediction
                results.append({
                    "property_name": prop_name,
                    "values": [0.0] * len(sequence),
                    "metadata": {
                        "error": str(e)
                    }
                })
        
        return {
            "sequence": sequence,
            "properties": results,
            "metadata": {
                "sequence_length": len(sequence)
            }
        }
    except Exception as e:
        logger.error(f"Error in property prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchPropertyResponse, responses={400: {"model": ErrorResponse}})
async def batch_predict_properties(request: BatchPropertyRequest):
    """
    Predict properties for multiple protein sequences.
    
    Parameters:
        request: Batch property prediction request
    
    Returns:
        Predicted property values for each sequence
    """
    try:
        results = {}
        errors = {}
        
        # Validate requested properties
        for prop in request.properties:
            if prop not in AVAILABLE_PROPERTIES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unknown property: {prop}. Available properties: {AVAILABLE_PROPERTIES}"
                )
        
        # Get property predictor
        predictor = get_property_predictor()
        
        for seq_id, sequence in request.sequences.items():
            try:
                # Process and validate sequence
                processed_seq = process_sequence(sequence)
                
                if not validate_sequence(processed_seq):
                    errors[seq_id] = "Invalid protein sequence"
                    continue
                
                # Generate predictions for each property
                property_results = []
                for prop_name in request.properties:
                    try:
                        prediction = predictor.predict_property(processed_seq, prop_name)
                        
                        # Convert numpy arrays to lists if needed
                        if isinstance(prediction, np.ndarray):
                            prediction = prediction.tolist()
                        
                        property_results.append({
                            "property_name": prop_name,
                            "values": prediction,
                            "metadata": {
                                "min": float(np.min(prediction)),
                                "max": float(np.max(prediction)),
                                "mean": float(np.mean(prediction))
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error predicting {prop_name} for {seq_id}: {e}")
                        # Add empty result for failed prediction
                        property_results.append({
                            "property_name": prop_name,
                            "values": [0.0] * len(processed_seq),
                            "metadata": {
                                "error": str(e)
                            }
                        })
                
                results[seq_id] = {
                    "sequence": processed_seq,
                    "properties": property_results,
                    "metadata": {
                        "sequence_length": len(processed_seq)
                    }
                }
            except Exception as e:
                errors[seq_id] = str(e)
        
        return {
            "results": results,
            "errors": errors,
            "total": len(request.sequences),
            "successful": len(results),
            "failed": len(errors)
        }
    except Exception as e:
        logger.error(f"Error in batch property prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=BatchPropertyResponse, responses={400: {"model": ErrorResponse}})
async def upload_file_properties(
    file: UploadFile = File(...),
    properties: List[str] = Query(["hydrophobicity", "secondary_structure"], description="Properties to predict"),
    max_sequences: int = Query(10, description="Maximum number of sequences to process")
):
    """
    Predict properties for sequences in uploaded file.
    
    Parameters:
        file: Uploaded file (FASTA, text, etc.)
        properties: Properties to predict
        max_sequences: Maximum number of sequences to process
    
    Returns:
        Predicted property values for each sequence
    """
    try:
        # Save uploaded file
        file_handler = get_file_handler()
        file_path = await file_handler.save_uploaded_file(file.file, file.filename)
        
        # Process file
        sequences = file_handler.process_file(file_path)
        
        # Limit number of sequences
        if len(sequences) > max_sequences:
            logger.warning(f"File contains {len(sequences)} sequences, limiting to {max_sequences}")
            sequences = dict(list(sequences.items())[:max_sequences])
        
        # Create batch request
        batch_request = BatchPropertyRequest(sequences=sequences, properties=properties)
        
        # Use batch endpoint
        result = await batch_predict_properties(batch_request)
        
        # Clean up file
        file_handler.cleanup(file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error processing uploaded file for property prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/uniprot", response_model=PropertyPredictionResponse, responses={400: {"model": ErrorResponse}})
async def uniprot_properties(
    uniprot_request: UniProtRequest,
    properties: List[str] = Query(["hydrophobicity", "secondary_structure"], description="Properties to predict")
):
    """
    Predict properties for a protein sequence from UniProt.
    
    Parameters:
        uniprot_request: UniProt ID request
        properties: Properties to predict
    
    Returns:
        Predicted property values for the sequence
    """
    try:
        # Get sequence from UniProt
        sequence = get_sequence_from_uniprot(uniprot_request.uniprot_id)
        
        if not sequence:
            raise HTTPException(status_code=404, detail=f"Sequence not found for UniProt ID: {uniprot_request.uniprot_id}")
        
        # Create property prediction request
        request = PropertyPredictionRequest(sequence=sequence, properties=properties)
        
        # Use standard prediction endpoint
        return await predict_properties(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting properties for UniProt sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available", response_model=List[str])
async def list_available_properties():
    """
    List all available properties that can be predicted.
    
    Returns:
        List of available property names
    """
    return AVAILABLE_PROPERTIES