from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Body, Depends
from typing import Dict, List, Optional, Union, Any
import logging
import numpy as np

from app.api.models.requests import (
    SequenceRequest, 
    BatchSequenceRequest, 
    UniProtRequest
)
from app.api.models.responses import (
    EmbeddingResponse, 
    BatchEmbeddingResponse, 
    ErrorResponse
)
from app.core.embedding import get_embedding_model
from app.services.sequence_utils import validate_sequence, process_sequence, get_sequence_from_uniprot
from app.services.file_handling import get_file_handler

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/generate", response_model=EmbeddingResponse, responses={400: {"model": ErrorResponse}})
async def generate_embedding(sequence_request: SequenceRequest):
    """
    Generate embeddings for a protein sequence.
    
    Parameters:
        sequence_request: Protein sequence request object
    
    Returns:
        Embedding vectors for the sequence
    """
    try:
        # Get sequence
        sequence = process_sequence(sequence_request.sequence)
        
        # Validate sequence
        if not validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Generate embedding
        embedding = embedding_model.get_embedding(sequence)
        
        # Convert numpy arrays to lists
        serializable_embedding = {
            'sequence': embedding['sequence'].tolist() if isinstance(embedding['sequence'], np.ndarray) else embedding['sequence'],
            'residue': embedding['residue'].tolist() if isinstance(embedding['residue'], np.ndarray) else embedding['residue']
        }
        
        return {
            "sequence": sequence,
            "embedding": serializable_embedding,
            "metadata": {
                "sequence_length": len(sequence),
                "embedding_dimensions": {
                    "sequence": len(serializable_embedding['sequence']),
                    "residue": len(serializable_embedding['residue'][0]) if serializable_embedding['residue'] else 0
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchEmbeddingResponse, responses={400: {"model": ErrorResponse}})
async def batch_embeddings(batch_request: BatchSequenceRequest):
    """
    Generate embeddings for multiple protein sequences.
    
    Parameters:
        batch_request: Batch of protein sequences
    
    Returns:
        Embedding vectors for each sequence
    """
    try:
        results = {}
        errors = {}
        
        for seq_id, sequence in batch_request.sequences.items():
            try:
                # Process and validate sequence
                processed_seq = process_sequence(sequence)
                
                if not validate_sequence(processed_seq):
                    errors[seq_id] = "Invalid protein sequence"
                    continue
                
                # Get embedding model
                embedding_model = get_embedding_model()
                
                # Generate embedding
                embedding = embedding_model.get_embedding(processed_seq)
                
                # Convert numpy arrays to lists
                serializable_embedding = {
                    'sequence': embedding['sequence'].tolist() if isinstance(embedding['sequence'], np.ndarray) else embedding['sequence'],
                    'residue': embedding['residue'].tolist() if isinstance(embedding['residue'], np.ndarray) else embedding['residue']
                }
                
                results[seq_id] = {
                    "sequence": processed_seq,
                    "embedding": serializable_embedding,
                    "metadata": {
                        "sequence_length": len(processed_seq),
                        "embedding_dimensions": {
                            "sequence": len(serializable_embedding['sequence']),
                            "residue": len(serializable_embedding['residue'][0]) if serializable_embedding['residue'] else 0
                        }
                    }
                }
            except Exception as e:
                errors[seq_id] = str(e)
        
        return {
            "results": results,
            "errors": errors,
            "total": len(batch_request.sequences),
            "successful": len(results),
            "failed": len(errors)
        }
    except Exception as e:
        logger.error(f"Error in batch embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=BatchEmbeddingResponse, responses={400: {"model": ErrorResponse}})
async def upload_file_embeddings(
    file: UploadFile = File(...),
    max_sequences: int = Query(10, description="Maximum number of sequences to process")
):
    """
    Generate embeddings for sequences in uploaded file.
    
    Parameters:
        file: Uploaded file (FASTA, text, etc.)
        max_sequences: Maximum number of sequences to process
    
    Returns:
        Embedding vectors for each sequence
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
            # Take first max_sequences
            sequences = dict(list(sequences.items())[:max_sequences])
        
        # Create batch request
        batch_request = BatchSequenceRequest(sequences=sequences)
        
        # Use batch endpoint
        result = await batch_embeddings(batch_request)
        
        # Clean up file
        file_handler.cleanup(file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/uniprot", response_model=EmbeddingResponse, responses={400: {"model": ErrorResponse}})
async def uniprot_embedding(uniprot_request: UniProtRequest):
    """
    Generate embeddings for a protein sequence from UniProt.
    
    Parameters:
        uniprot_request: UniProt ID request
    
    Returns:
        Embedding vectors for the sequence
    """
    try:
        # Get sequence from UniProt
        sequence = get_sequence_from_uniprot(uniprot_request.uniprot_id)
        
        if not sequence:
            raise HTTPException(status_code=404, detail=f"Sequence not found for UniProt ID: {uniprot_request.uniprot_id}")
        
        # Create sequence request
        sequence_request = SequenceRequest(sequence=sequence)
        
        # Use standard embedding endpoint
        return await generate_embedding(sequence_request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating UniProt embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reduce", responses={400: {"model": ErrorResponse}})
async def reduce_embedding(
    sequence: str = Body(..., embed=True, description="Protein sequence"),
    method: str = Query("pca", description="Dimensionality reduction method (pca, tsne, umap)"),
    dimensions: int = Query(2, description="Number of dimensions to reduce to")
):
    """
    Generate reduced dimensionality embeddings for visualization.
    
    Parameters:
        sequence: Protein sequence
        method: Dimensionality reduction method
        dimensions: Number of dimensions
    
    Returns:
        Reduced embedding vectors
    """
    try:
        # Process and validate sequence
        processed_seq = process_sequence(sequence)
        
        if not validate_sequence(processed_seq):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Validate method
        valid_methods = ["pca", "tsne", "umap"]
        if method not in valid_methods:
            raise HTTPException(status_code=400, detail=f"Invalid method. Choose from: {valid_methods}")
        
        # Validate dimensions
        if dimensions < 2 or dimensions > 3:
            raise HTTPException(status_code=400, detail="Dimensions must be 2 or 3")
        
        # Get embedding model
        embedding_model = get_embedding_model()
        
        # Generate reduced embedding
        reduced = embedding_model.get_reduced_embedding(processed_seq, method, dimensions)
        
        # Convert to list for serialization
        reduced_list = reduced.tolist() if isinstance(reduced, np.ndarray) else reduced
        
        return {
            "sequence": processed_seq,
            "reduced_embedding": reduced_list,
            "metadata": {
                "method": method,
                "dimensions": dimensions,
                "sequence_length": len(processed_seq)
            }
        }
    except Exception as e:
        logger.error(f"Error reducing embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))