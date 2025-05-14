from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Any
import re

class SequenceRequest(BaseModel):
    """Request model for a single protein sequence."""
    sequence: str = Field(..., description="Protein sequence", min_length=5, max_length=10000)
    
    @validator('sequence')
    def validate_sequence_chars(cls, v):
        """Validate that the sequence contains valid amino acid characters."""
        v = v.strip().upper()
        # Remove whitespace
        v = re.sub(r'\s+', '', v)
        
        # Check for valid amino acids (with some allowed ambiguous characters)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYBZX*-")
        if not all(aa in valid_aa for aa in v):
            invalid_chars = set(v) - valid_aa
            raise ValueError(f"Invalid amino acid characters in sequence: {invalid_chars}")
        
        return v

class BatchSequenceRequest(BaseModel):
    """Request model for multiple protein sequences."""
    sequences: Dict[str, str] = Field(..., description="Dictionary mapping sequence IDs to sequences")
    
    @validator('sequences')
    def validate_sequences(cls, v):
        """Validate that no more than 50 sequences are provided."""
        if len(v) > 50:
            raise ValueError(f"Maximum of 50 sequences allowed, got {len(v)}")
        return v

class UniProtRequest(BaseModel):
    """Request model for UniProt ID."""
    uniprot_id: str = Field(..., description="UniProt ID", regex=r"^[A-Z0-9]{6,10}$")

class PropertyPredictionRequest(BaseModel):
    """Request model for property prediction."""
    sequence: str = Field(..., description="Protein sequence", min_length=5, max_length=10000)
    properties: List[str] = Field(
        ["hydrophobicity", "charge", "secondary_structure"], 
        description="Properties to predict"
    )

class BatchPropertyRequest(BaseModel):
    """Request model for batch property prediction."""
    sequences: Dict[str, str] = Field(..., description="Dictionary mapping sequence IDs to sequences")
    properties: List[str] = Field(
        ["hydrophobicity", "charge", "secondary_structure"], 
        description="Properties to predict"
    )

class ClassificationRequest(BaseModel):
    """Request model for classification."""
    sequence: str = Field(..., description="Protein sequence", min_length=5, max_length=10000)
    classification_types: List[str] = Field(
        ["subcellular_localization", "antimicrobial_potential", "toxicity"], 
        description="Classification types to predict"
    )

class BatchClassificationRequest(BaseModel):
    """Request model for batch classification."""
    sequences: Dict[str, str] = Field(..., description="Dictionary mapping sequence IDs to sequences")
    classification_types: List[str] = Field(
        ["subcellular_localization", "antimicrobial_potential", "toxicity"], 
        description="Classification types to predict"
    )

class StabilityConditions(BaseModel):
    """Model for stability prediction conditions."""
    temperature: float = Field(25.0, description="Temperature in Celsius")
    ph: float = Field(7.0, description="pH value")
    ionic_strength: float = Field(150.0, description="Ionic strength in mM")
    pressure: float = Field(1.0, description="Pressure in atm")
    denaturant_conc: float = Field(0.0, description="Denaturant concentration in M")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 100:
            raise ValueError("Temperature must be between 0 and 100°C")
        return v
    
    @validator('ph')
    def validate_ph(cls, v):
        if v < 0 or v > 14:
            raise ValueError("pH must be between 0 and 14")
        return v

class StabilityRequest(BaseModel):
    """Request model for stability prediction."""
    sequence: str = Field(..., description="Protein sequence", min_length=5, max_length=10000)
    conditions: Optional[StabilityConditions] = Field(None, description="Stability conditions")

class BatchStabilityRequest(BaseModel):
    """Request model for batch stability prediction."""
    sequences: Dict[str, str] = Field(..., description="Dictionary mapping sequence IDs to sequences")
    conditions: Optional[StabilityConditions] = Field(None, description="Stability conditions")

class StructurePredictionRequest(BaseModel):
    """Request model for structure prediction."""
    sequence: str = Field(..., description="Protein sequence", min_length=10, max_length=1500)
    use_relaxation: bool = Field(True, description="Whether to use Amber relaxation")

class PropertyMappingRequest(BaseModel):
    """Request model for mapping properties to structure."""
    pdb_path: str = Field(..., description="Path to PDB file")
    properties: Dict[str, List[float]] = Field(..., description="Dictionary mapping property names to per-residue values")