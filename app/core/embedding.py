import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
import pickle
import hashlib

from app.config import settings
from app.services.sequence_utils import validate_sequence, process_sequence

# Configure logging
logger = logging.getLogger(__name__)

# Try to import transformers, handle if not available
try:
    from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers package not available. Using fallback embedding method.")
    TRANSFORMERS_AVAILABLE = False

# ESM model constants
ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LENGTH = 1024

class ProteinEmbeddingModel:
    """
    Handles protein sequence embedding generation using pretrained models.
    Primary implementation uses ESM-2 (Evolutionary Scale Modeling) from Meta AI.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the embedding model."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.model_path = model_path or settings.EMBEDDING_MODEL_PATH
        self.tokenizer = None
        self.model = None
        self.cache = {}
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Using simplified embedding model due to missing dependencies")
            return
        
        try:
            logger.info(f"Loading protein embedding model from {ESM_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
            self.model = AutoModel.from_pretrained(ESM_MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Protein embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load protein embedding model: {e}")
            raise
    
    def _sequence_hash(self, sequence: str) -> str:
        """Generate a hash for the sequence for caching purposes."""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def _check_cache(self, sequence: str) -> Optional[np.ndarray]:
        """Check if embedding for sequence exists in cache."""
        if not settings.CACHE_ENABLED:
            return None
            
        sequence_hash = self._sequence_hash(sequence)
        cache_file = settings.CACHE_DIR / f"emb_{sequence_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
                return None
        return None
    
    def _save_to_cache(self, sequence: str, embedding: np.ndarray):
        """Save embedding to cache."""
        if not settings.CACHE_ENABLED:
            return
            
        sequence_hash = self._sequence_hash(sequence)
        cache_file = settings.CACHE_DIR / f"emb_{sequence_hash}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def get_embedding(self, sequence: str) -> np.ndarray:
        """
        Generate embedding for a protein sequence.
        
        Args:
            sequence: Amino acid sequence string
        
        Returns:
            Numpy array of embedding vectors
        """
        # Validate sequence
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Check cache
        cached_embedding = self._check_cache(sequence)
        if cached_embedding is not None:
            return cached_embedding
        
        # If transformers not available, use simplified embedding
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return self._simplified_embedding(sequence)
        
        # Generate embedding using ESM model
        try:
            # Handle long sequences
            if len(sequence) > MAX_SEQ_LENGTH:
                logger.warning(f"Sequence length ({len(sequence)}) exceeds model maximum ({MAX_SEQ_LENGTH}). Truncating.")
                sequence = sequence[:MAX_SEQ_LENGTH]
            
            # Tokenize and generate embedding
            with torch.no_grad():
                inputs = self.tokenizer(sequence, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Use the last hidden state of the [CLS] token as the sequence embedding
                sequence_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()
                
                # Get per-residue embeddings from the last hidden layer
                residue_embeddings = outputs.last_hidden_state[0, 1:-1].cpu().numpy()
                
                # Combine into a single embedding structure
                embedding = {
                    'sequence': sequence_embedding,  # Global sequence embedding
                    'residue': residue_embeddings,   # Per-residue embeddings
                }
                
                # Cache the result
                self._save_to_cache(sequence, embedding)
                
                return embedding
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._simplified_embedding(sequence)
    
    def _simplified_embedding(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Fallback method to generate simple embeddings when the main model is unavailable.
        Uses a simple one-hot encoding approach.
        """
        logger.info("Using simplified embedding method")
        
        # Define amino acid vocabulary
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        # Create one-hot encoding for each residue
        seq_length = len(sequence)
        embedding_dim = len(amino_acids)
        residue_embeddings = np.zeros((seq_length, embedding_dim))
        
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                residue_embeddings[i, aa_to_idx[aa]] = 1.0
        
        # Create sequence-level embedding by averaging residue embeddings
        sequence_embedding = np.mean(residue_embeddings, axis=0)
        
        return {
            'sequence': sequence_embedding,
            'residue': residue_embeddings
        }
    
    def get_reduced_embedding(self, sequence: str, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """
        Generate dimensionality-reduced embedding for visualization.
        
        Args:
            sequence: Amino acid sequence string
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components for reduction
            
        Returns:
            Reduced embedding vectors
        """
        # Get full embedding
        embedding = self.get_embedding(sequence)
        residue_embeddings = embedding['residue']
        
        # Apply dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components)
            except ImportError:
                logger.warning("UMAP not available, falling back to PCA")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Perform reduction
        reduced_embeddings = reducer.fit_transform(residue_embeddings)
        
        return reduced_embeddings


# Singleton instance
_model_instance = None

def get_embedding_model() -> ProteinEmbeddingModel:
    """Get or create the embedding model singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ProteinEmbeddingModel()
    return _model_instance