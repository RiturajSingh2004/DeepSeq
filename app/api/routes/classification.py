from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body, Depends
from typing import Dict, List, Optional, Union, Any
import logging

from app.api.models.requests import (
    ClassificationRequest,
    BatchClassificationRequest,
    UniProtRequest
)
from app.api.models.responses import (
    ClassificationResponse,
    BatchClassificationResponse,
    ErrorResponse
)
from app.core.classification import get_classifier, CLASSIFICATION_TYPES, CLASSIFICATION_CLASSES
from app.services.sequence_utils import validate_sequence, process_sequence, get_sequence_from_uniprot
from app.services.file_handling import get_file_handler

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict", response_model=ClassificationResponse, responses={400: {"model": ErrorResponse}})
async def classify_sequence(request: ClassificationRequest):
    """
    Classify a protein sequence for various functional properties.
    
    Parameters:
        request: Classification request with sequence and classification types
    
    Returns:
        Classification results with probabilities
    """
    try:
        # Process and validate sequence
        sequence = process_sequence(request.sequence)
        
        if not validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Validate requested classification types
        for cls_type in request.classification_types:
            if cls_type not in CLASSIFICATION_TYPES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unknown classification type: {cls_type}. Available types: {CLASSIFICATION_TYPES}"
                )
        
        # Get classifier
        classifier = get_classifier()
        
        # Generate classifications
        results = []
        for cls_type in request.classification_types:
            try:
                prediction = classifier.classify(sequence, cls_type)
                
                results.append({
                    "classification_type": cls_type,
                    "predictions": prediction,
                    "metadata": {
                        "possible_classes": CLASSIFICATION_CLASSES.get(cls_type, []),
                        "confidence": sum(prediction.values()) / len(prediction)  # Average confidence
                    }
                })
            except Exception as e:
                logger.error(f"Error classifying {cls_type}: {e}")
                # Add empty result for failed classification
                results.append({
                    "classification_type": cls_type,
                    "predictions": {cls: 0.0 for cls in CLASSIFICATION_CLASSES.get(cls_type, [])},
                    "metadata": {
                        "error": str(e)
                    }
                })
        
        return {
            "sequence": sequence,
            "classifications": results,
            "metadata": {
                "sequence_length": len(sequence)
            }
        }
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchClassificationResponse, responses={400: {"model": ErrorResponse}})
async def batch_classify_sequences(request: BatchClassificationRequest):
    """
    Classify multiple protein sequences.
    
    Parameters:
        request: Batch classification request
    
    Returns:
        Classification results for each sequence
    """
    try:
        results = {}
        errors = {}
        
        # Validate requested classification types
        for cls_type in request.classification_types:
            if cls_type not in CLASSIFICATION_TYPES:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unknown classification type: {cls_type}. Available types: {CLASSIFICATION_TYPES}"
                )
        
        # Get classifier
        classifier = get_classifier()
        
        for seq_id, sequence in request.sequences.items():
            try:
                # Process and validate sequence
                processed_seq = process_sequence(sequence)
                
                if not validate_sequence(processed_seq):
                    errors[seq_id] = "Invalid protein sequence"
                    continue
                
                # Generate classifications for each type
                classification_results = []
                for cls_type in request.classification_types:
                    try:
                        prediction = classifier.classify(processed_seq, cls_type)
                        
                        classification_results.append({
                            "classification_type": cls_type,
                            "predictions": prediction,
                            "metadata": {
                                "possible_classes": CLASSIFICATION_CLASSES.get(cls_type, []),
                                "confidence": sum(prediction.values()) / len(prediction)  # Average confidence
                            }
                        })
                    except Exception as e:
                        logger.error(f"Error classifying {cls_type} for {seq_id}: {e}")
                        # Add empty result for failed classification
                        classification_results.append({
                            "classification_type": cls_type,
                            "predictions": {cls: 0.0 for cls in CLASSIFICATION_CLASSES.get(cls_type, [])},
                            "metadata": {
                                "error": str(e)
                            }
                        })
                
                results[seq_id] = {
                    "sequence": processed_seq,
                    "classifications": classification_results,
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
        logger.error(f"Error in batch classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=BatchClassificationResponse, responses={400: {"model": ErrorResponse}})
async def upload_file_classification(
    file: UploadFile = File(...),
    classification_types: List[str] = Query(
        ["subcellular_localization", "antimicrobial_potential"], 
        description="Classification types to predict"
    ),
    max_sequences: int = Query(10, description="Maximum number of sequences to process")
):
    """
    Classify sequences in uploaded file.
    
    Parameters:
        file: Uploaded file (FASTA, text, etc.)
        classification_types: Classification types to predict
        max_sequences: Maximum number of sequences to process
    
    Returns:
        Classification results for each sequence
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
        batch_request = BatchClassificationRequest(
            sequences=sequences, 
            classification_types=classification_types
        )
        
        # Use batch endpoint
        result = await batch_classify_sequences(batch_request)
        
        # Clean up file
        file_handler.cleanup(file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error processing uploaded file for classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/uniprot", response_model=ClassificationResponse, responses={400: {"model": ErrorResponse}})
async def uniprot_classification(
    uniprot_request: UniProtRequest,
    classification_types: List[str] = Query(
        ["subcellular_localization", "antimicrobial_potential"], 
        description="Classification types to predict"
    )
):
    """
    Classify a protein sequence from UniProt.
    
    Parameters:
        uniprot_request: UniProt ID request
        classification_types: Classification types to predict
    
    Returns:
        Classification results for the sequence
    """
    try:
        # Get sequence from UniProt
        sequence = get_sequence_from_uniprot(uniprot_request.uniprot_id)
        
        if not sequence:
            raise HTTPException(status_code=404, detail=f"Sequence not found for UniProt ID: {uniprot_request.uniprot_id}")
        
        # Create classification request
        request = ClassificationRequest(sequence=sequence, classification_types=classification_types)
        
        # Use standard classification endpoint
        return await classify_sequence(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error classifying UniProt sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available", response_model=Dict[str, List[str]])
async def list_classification_types():
    """
    List all available classification types and their possible classes.
    
    Returns:
        Dictionary mapping classification types to their possible classes
    """
    return CLASSIFICATION_CLASSES