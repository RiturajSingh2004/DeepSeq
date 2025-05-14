from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body, Depends
from typing import Dict, List, Optional, Union, Any
import logging

from app.api.models.requests import (
    StabilityRequest,
    BatchStabilityRequest,
    UniProtRequest,
    StabilityConditions
)
from app.api.models.responses import (
    StabilityResponse,
    BatchStabilityResponse,
    ErrorResponse
)
from app.core.stability import get_stability_estimator
from app.services.sequence_utils import validate_sequence, process_sequence, get_sequence_from_uniprot
from app.services.file_handling import get_file_handler

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict", response_model=StabilityResponse, responses={400: {"model": ErrorResponse}})
async def predict_stability(request: StabilityRequest):
    """
    Predict protein stability under specified conditions.
    
    Parameters:
        request: Stability prediction request with sequence and conditions
    
    Returns:
        Stability prediction results
    """
    try:
        # Process and validate sequence
        sequence = process_sequence(request.sequence)
        
        if not validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Get stability estimator
        estimator = get_stability_estimator()
        
        # Get conditions
        conditions = request.conditions.dict() if request.conditions else None
        
        # Predict stability
        prediction = estimator.predict_stability(sequence, conditions)
        
        # Format response
        stability_result = {
            "melting_temp": prediction.get("melting_temp", 0.0),
            "free_energy": prediction.get("free_energy", 0.0),
            "ph_stability": prediction.get("ph_stability", {"min_ph": 0.0, "max_ph": 0.0}),
            "confidence": prediction.get("confidence", 0.0)
        }
        
        # Get conditions used (either from request or defaults)
        used_conditions = conditions or {
            "temperature": 25.0,
            "ph": 7.0,
            "ionic_strength": 150.0,
            "pressure": 1.0,
            "denaturant_conc": 0.0
        }
        
        return {
            "sequence": sequence,
            "conditions": used_conditions,
            "stability": stability_result,
            "metadata": {
                "sequence_length": len(sequence)
            }
        }
    except Exception as e:
        logger.error(f"Error in stability prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchStabilityResponse, responses={400: {"model": ErrorResponse}})
async def batch_predict_stability(request: BatchStabilityRequest):
    """
    Predict stability for multiple protein sequences.
    
    Parameters:
        request: Batch stability prediction request
    
    Returns:
        Stability prediction results for each sequence
    """
    try:
        results = {}
        errors = {}
        
        # Get stability estimator
        estimator = get_stability_estimator()
        
        # Get conditions
        conditions = request.conditions.dict() if request.conditions else None
        
        # Get conditions used (either from request or defaults)
        used_conditions = conditions or {
            "temperature": 25.0,
            "ph": 7.0,
            "ionic_strength": 150.0,
            "pressure": 1.0,
            "denaturant_conc": 0.0
        }
        
        for seq_id, sequence in request.sequences.items():
            try:
                # Process and validate sequence
                processed_seq = process_sequence(sequence)
                
                if not validate_sequence(processed_seq):
                    errors[seq_id] = "Invalid protein sequence"
                    continue
                
                # Predict stability
                prediction = estimator.predict_stability(processed_seq, conditions)
                
                # Format result
                stability_result = {
                    "melting_temp": prediction.get("melting_temp", 0.0),
                    "free_energy": prediction.get("free_energy", 0.0),
                    "ph_stability": prediction.get("ph_stability", {"min_ph": 0.0, "max_ph": 0.0}),
                    "confidence": prediction.get("confidence", 0.0)
                }
                
                results[seq_id] = {
                    "sequence": processed_seq,
                    "conditions": used_conditions,
                    "stability": stability_result,
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
        logger.error(f"Error in batch stability prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=BatchStabilityResponse, responses={400: {"model": ErrorResponse}})
async def upload_file_stability(
    file: UploadFile = File(...),
    conditions: Optional[StabilityConditions] = None,
    max_sequences: int = Query(10, description="Maximum number of sequences to process")
):
    """
    Predict stability for sequences in uploaded file.
    
    Parameters:
        file: Uploaded file (FASTA, text, etc.)
        conditions: Stability conditions
        max_sequences: Maximum number of sequences to process
    
    Returns:
        Stability prediction results for each sequence
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
        batch_request = BatchStabilityRequest(
            sequences=sequences, 
            conditions=conditions
        )
        
        # Use batch endpoint
        result = await batch_predict_stability(batch_request)
        
        # Clean up file
        file_handler.cleanup(file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error processing uploaded file for stability prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/uniprot", response_model=StabilityResponse, responses={400: {"model": ErrorResponse}})
async def uniprot_stability(
    uniprot_request: UniProtRequest,
    conditions: Optional[StabilityConditions] = None
):
    """
    Predict stability for a protein sequence from UniProt.
    
    Parameters:
        uniprot_request: UniProt ID request
        conditions: Stability conditions
    
    Returns:
        Stability prediction results for the sequence
    """
    try:
        # Get sequence from UniProt
        sequence = get_sequence_from_uniprot(uniprot_request.uniprot_id)
        
        if not sequence:
            raise HTTPException(status_code=404, detail=f"Sequence not found for UniProt ID: {uniprot_request.uniprot_id}")
        
        # Create stability request
        request = StabilityRequest(sequence=sequence, conditions=conditions)
        
        # Use standard stability endpoint
        return await predict_stability(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting stability for UniProt sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))