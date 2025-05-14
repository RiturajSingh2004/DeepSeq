from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error details")

class EmbeddingMetadata(BaseModel):
    """Metadata for embedding responses."""
    sequence_length: int = Field(..., description="Length of the protein sequence")
    embedding_dimensions: Dict[str, int] = Field(..., description="Dimensions of embedding vectors")

class EmbeddingData(BaseModel):
    """Embedding data structure."""
    sequence: List[float] = Field(..., description="Global sequence embedding vector")
    residue: List[List[float]] = Field(..., description="Per-residue embedding vectors")

class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    sequence: str = Field(..., description="Protein sequence")
    embedding: EmbeddingData = Field(..., description="Embedding vectors")
    metadata: EmbeddingMetadata = Field(..., description="Embedding metadata")

class BatchEmbeddingResult(BaseModel):
    """Result for a single sequence in a batch."""
    sequence: str = Field(..., description="Protein sequence")
    embedding: EmbeddingData = Field(..., description="Embedding vectors")
    metadata: EmbeddingMetadata = Field(..., description="Embedding metadata")

class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""
    results: Dict[str, BatchEmbeddingResult] = Field(..., description="Results for each sequence")
    errors: Dict[str, str] = Field(..., description="Errors for failed sequences")
    total: int = Field(..., description="Total number of sequences")
    successful: int = Field(..., description="Number of successful embeddings")
    failed: int = Field(..., description="Number of failed embeddings")

class PropertyPredictionResult(BaseModel):
    """Result for a single property prediction."""
    property_name: str = Field(..., description="Name of the predicted property")
    values: List[float] = Field(..., description="Predicted values for each residue")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata for the prediction")

class PropertyPredictionResponse(BaseModel):
    """Response model for property prediction."""
    sequence: str = Field(..., description="Protein sequence")
    properties: List[PropertyPredictionResult] = Field(..., description="Predicted properties")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

class BatchPropertyResult(BaseModel):
    """Result for a single sequence in a batch property prediction."""
    sequence: str = Field(..., description="Protein sequence")
    properties: List[PropertyPredictionResult] = Field(..., description="Predicted properties")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

class BatchPropertyResponse(BaseModel):
    """Response model for batch property prediction."""
    results: Dict[str, BatchPropertyResult] = Field(..., description="Results for each sequence")
    errors: Dict[str, str] = Field(..., description="Errors for failed sequences")
    total: int = Field(..., description="Total number of sequences")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")

class ClassificationResult(BaseModel):
    """Result for a single classification."""
    classification_type: str = Field(..., description="Type of classification")
    predictions: Dict[str, float] = Field(..., description="Predicted class probabilities")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata for the classification")

class ClassificationResponse(BaseModel):
    """Response model for classification."""
    sequence: str = Field(..., description="Protein sequence")
    classifications: List[ClassificationResult] = Field(..., description="Classification results")
    metadata: Dict[str, Any] = Field(..., description="Classification metadata")

class BatchClassificationResult(BaseModel):
    """Result for a single sequence in a batch classification."""
    sequence: str = Field(..., description="Protein sequence")
    classifications: List[ClassificationResult] = Field(..., description="Classification results")
    metadata: Dict[str, Any] = Field(..., description="Classification metadata")

class BatchClassificationResponse(BaseModel):
    """Response model for batch classification."""
    results: Dict[str, BatchClassificationResult] = Field(..., description="Results for each sequence")
    errors: Dict[str, str] = Field(..., description="Errors for failed sequences")
    total: int = Field(..., description="Total number of sequences")
    successful: int = Field(..., description="Number of successful classifications")
    failed: int = Field(..., description="Number of failed classifications")

class PHStabilityRange(BaseModel):
    """pH stability range."""
    min_ph: float = Field(..., description="Minimum stable pH")
    max_ph: float = Field(..., description="Maximum stable pH")

class StabilityResult(BaseModel):
    """Stability prediction result."""
    melting_temp: float = Field(..., description="Predicted melting temperature in Celsius")
    free_energy: float = Field(..., description="Predicted folding free energy in kcal/mol")
    ph_stability: PHStabilityRange = Field(..., description="pH range where protein is stable")
    confidence: float = Field(..., description="Confidence score of the prediction (0-1)")

class StabilityResponse(BaseModel):
    """Response model for stability prediction."""
    sequence: str = Field(..., description="Protein sequence")
    conditions: Dict[str, float] = Field(..., description="Stability conditions used")
    stability: StabilityResult = Field(..., description="Stability prediction results")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

class BatchStabilityResult(BaseModel):
    """Result for a single sequence in a batch stability prediction."""
    sequence: str = Field(..., description="Protein sequence")
    conditions: Dict[str, float] = Field(..., description="Stability conditions used")
    stability: StabilityResult = Field(..., description="Stability prediction results")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")

class BatchStabilityResponse(BaseModel):
    """Response model for batch stability prediction."""
    results: Dict[str, BatchStabilityResult] = Field(..., description="Results for each sequence")
    errors: Dict[str, str] = Field(..., description="Errors for failed sequences")
    total: int = Field(..., description="Total number of sequences")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")

class StructureConfidence(BaseModel):
    """Structure prediction confidence scores."""
    plddt: Union[List[float], float] = Field(..., description="Per-residue local distance difference test scores")
    plddt_mean: float = Field(..., description="Mean pLDDT score")
    ptm: float = Field(..., description="Predicted TM-score")
    iptm: float = Field(..., description="Interface predicted TM-score")

class StructurePredictionResponse(BaseModel):
    """Response model for structure prediction."""
    sequence: str = Field(..., description="Protein sequence")
    pdb_path: str = Field(..., description="Path to PDB file")
    confidence: StructureConfidence = Field(..., description="Prediction confidence scores")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")