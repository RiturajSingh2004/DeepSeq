from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body, Depends
from fastapi.responses import FileResponse
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path

from app.api.models.requests import (
    StructurePredictionRequest,
    PropertyMappingRequest,
    UniProtRequest
)
from app.api.models.responses import (
    StructurePredictionResponse,
    ErrorResponse
)
from app.core.structure import get_structure_predictor
from app.core.property_prediction import get_property_predictor, AVAILABLE_PROPERTIES
from app.services.sequence_utils import validate_sequence, process_sequence, get_sequence_from_uniprot
from app.services.file_handling import get_file_handler

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/predict", response_model=StructurePredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_structure(request: StructurePredictionRequest):
    """
    Predict 3D structure of a protein sequence using AlphaFold integration.
    
    Parameters:
        request: Structure prediction request with sequence and options
    
    Returns:
        Structure prediction results including path to PDB file
    """
    try:
        # Process and validate sequence
        sequence = process_sequence(request.sequence)
        
        if not validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Check sequence length limits
        if len(sequence) < 10:
            raise HTTPException(status_code=400, detail="Sequence too short for structure prediction (minimum 10 residues)")
            
        if len(sequence) > 1500:
            raise HTTPException(status_code=400, detail="Sequence too long for structure prediction (maximum 1500 residues)")
        
        # Get structure predictor
        predictor = get_structure_predictor()
        
        # Predict structure
        result = predictor.predict_structure(sequence, request.use_relaxation)
        
        # Format response
        return {
            "sequence": sequence,
            "pdb_path": result["pdb_path"],
            "confidence": result["confidence"],
            "metadata": result["metadata"]
        }
    except Exception as e:
        logger.error(f"Error in structure prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{pdb_id}")
async def download_pdb(pdb_id: str):
    """
    Download a PDB file.
    
    Parameters:
        pdb_id: ID of the PDB file to download
    
    Returns:
        PDB file
    """
    try:
        # Validate PDB ID to prevent path traversal
        if not pdb_id.isalnum() and not all(c in "._-" for c in pdb_id if not c.isalnum()):
            raise HTTPException(status_code=400, detail="Invalid PDB ID")
        
        # Get structure predictor to determine results directory
        predictor = get_structure_predictor()
        pdb_path = predictor.results_dir / f"{pdb_id}.pdb"
        
        if not pdb_path.exists():
            raise HTTPException(status_code=404, detail=f"PDB file not found: {pdb_id}")
        
        return FileResponse(
            path=pdb_path,
            filename=f"{pdb_id}.pdb",
            media_type="chemical/x-pdb"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error downloading PDB file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/map-properties")
async def map_properties_to_structure(request: PropertyMappingRequest):
    """
    Map per-residue properties to B-factor column in PDB for visualization.
    
    Parameters:
        request: Property mapping request with PDB path and properties
    
    Returns:
        Path to new PDB with properties mapped
    """
    try:
        # Validate PDB path
        pdb_path = Path(request.pdb_path)
        if not pdb_path.exists():
            raise HTTPException(status_code=404, detail=f"PDB file not found: {pdb_path}")
        
        # Get structure predictor
        predictor = get_structure_predictor()
        
        # Map properties to structure
        mapped_pdb_path = predictor.map_properties_to_structure(str(pdb_path), request.properties)
        
        return {
            "original_pdb_path": str(pdb_path),
            "mapped_pdb_path": mapped_pdb_path,
            "mapped_properties": list(request.properties.keys())
        }
    except Exception as e:
        logger.error(f"Error mapping properties to structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-and-map", response_model=StructurePredictionResponse, responses={400: {"model": ErrorResponse}})
async def predict_structure_with_properties(
    sequence: str = Body(..., embed=True),
    property_name: str = Query("hydrophobicity", description="Property to map to structure"),
    use_relaxation: bool = Query(True, description="Whether to use Amber relaxation")
):
    """
    Predict structure and map a property to the structure in one step.
    
    Parameters:
        sequence: Protein sequence
        property_name: Property to map to structure
        use_relaxation: Whether to use Amber relaxation
    
    Returns:
        Structure prediction results with property mapping
    """
    try:
        # Validate sequence
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise HTTPException(status_code=400, detail="Invalid protein sequence")
        
        # Validate property
        if property_name not in AVAILABLE_PROPERTIES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown property: {property_name}. Available properties: {AVAILABLE_PROPERTIES}"
            )
        
        # Create structure prediction request
        structure_request = StructurePredictionRequest(
            sequence=sequence,
            use_relaxation=use_relaxation
        )
        
        # Predict structure
        structure_result = await predict_structure(structure_request)
        
        # Get property predictor
        property_predictor = get_property_predictor()
        
        # Predict property
        property_values = property_predictor.predict_property(sequence, property_name)
        
        # Map property to structure
        mapping_request = PropertyMappingRequest(
            pdb_path=structure_result["pdb_path"],
            properties={property_name: property_values.tolist() if hasattr(property_values, 'tolist') else property_values}
        )
        
        mapping_result = await map_properties_to_structure(mapping_request)
        
        # Update structure result with mapped PDB path
        structure_result["mapped_pdb_path"] = mapping_result["mapped_pdb_path"]
        structure_result["mapped_property"] = property_name
        
        return structure_result
    except Exception as e:
        logger.error(f"Error in combined structure prediction and property mapping: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/uniprot", response_model=StructurePredictionResponse, responses={400: {"model": ErrorResponse}})
async def uniprot_structure(
    uniprot_request: UniProtRequest,
    use_relaxation: bool = Query(True, description="Whether to use Amber relaxation")
):
    """
    Predict structure for a protein sequence from UniProt.
    
    Parameters:
        uniprot_request: UniProt ID request
        use_relaxation: Whether to use Amber relaxation
    
    Returns:
        Structure prediction results for the sequence
    """
    try:
        # Get sequence from UniProt
        sequence = get_sequence_from_uniprot(uniprot_request.uniprot_id)
        
        if not sequence:
            raise HTTPException(status_code=404, detail=f"Sequence not found for UniProt ID: {uniprot_request.uniprot_id}")
        
        # Create structure prediction request
        request = StructurePredictionRequest(
            sequence=sequence,
            use_relaxation=use_relaxation
        )
        
        # Use standard structure prediction endpoint
        return await predict_structure(request)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting structure for UniProt sequence: {e}")
        raise HTTPException(status_code=500, detail=str(e))