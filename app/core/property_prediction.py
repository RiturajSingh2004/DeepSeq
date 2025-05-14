import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from pathlib import Path
import pickle
import hashlib

from app.config import settings
from app.services.sequence_utils import validate_sequence, process_sequence
from app.core.embedding import get_embedding_model

# Configure logging
logger = logging.getLogger(__name__)

# List of properties this module can predict
AVAILABLE_PROPERTIES = [
    "hydrophobicity",
    "charge",
    "secondary_structure",
    "solvent_accessibility",
    "disorder",
    "binding_sites",
    "flexibility"
]

class ProteinPropertyPredictor:
    """
    Predicts various per-residue properties of protein sequences based on embeddings.
    Uses pretrained models to map from embedding space to property values.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize property prediction models."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.model_path = model_path or settings.PROPERTY_MODEL_PATH
        self.embedding_model = get_embedding_model()
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load the pretrained property prediction models."""
        try:
            for property_name in AVAILABLE_PROPERTIES:
                model_file = self.model_path / f"{property_name}_model.pt"
                
                if model_file.exists():
                    logger.info(f"Loading {property_name} prediction model")
                    self.models[property_name] = self._load_specific_model(model_file)
                else:
                    logger.warning(f"Model file for {property_name} not found at {model_file}")
                    # Use simplified model as fallback
                    self.models[property_name] = None
            
            logger.info(f"Loaded {len(self.models)} property prediction models")
        except Exception as e:
            logger.error(f"Failed to load property prediction models: {e}")
    
    def _load_specific_model(self, model_path: Path) -> Any:
        """Load a specific property prediction model."""
        try:
            # Simple MLP model for property prediction
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def _sequence_hash(self, sequence: str, property_name: str) -> str:
        """Generate a hash for caching purposes."""
        return hashlib.md5(f"{sequence}_{property_name}".encode()).hexdigest()
    
    def _check_cache(self, sequence: str, property_name: str) -> Optional[np.ndarray]:
        """Check if prediction exists in cache."""
        if not settings.CACHE_ENABLED:
            return None
            
        sequence_hash = self._sequence_hash(sequence, property_name)
        cache_file = settings.CACHE_DIR / f"prop_{sequence_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached prediction: {e}")
                return None
        return None
    
    def _save_to_cache(self, sequence: str, property_name: str, prediction: np.ndarray):
        """Save prediction to cache."""
        if not settings.CACHE_ENABLED:
            return
            
        sequence_hash = self._sequence_hash(sequence, property_name)
        cache_file = settings.CACHE_DIR / f"prop_{sequence_hash}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(prediction, f)
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
    
    def predict_property(self, sequence: str, property_name: str) -> np.ndarray:
        """
        Predict a specific property for each residue in the sequence.
        
        Args:
            sequence: Amino acid sequence string
            property_name: Name of the property to predict
            
        Returns:
            Array of property values for each residue
        """
        # Validate inputs
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
            
        if property_name not in AVAILABLE_PROPERTIES:
            raise ValueError(f"Unknown property: {property_name}. Available properties: {AVAILABLE_PROPERTIES}")
        
        # Check cache
        cached_prediction = self._check_cache(sequence, property_name)
        if cached_prediction is not None:
            return cached_prediction
        
        # Get sequence embedding
        embedding = self.embedding_model.get_embedding(sequence)
        residue_embeddings = embedding['residue']
        
        # Get appropriate model
        model = self.models.get(property_name)
        
        # Use pretrained model if available, otherwise use simplified prediction
        if model is not None:
            try:
                with torch.no_grad():
                    # Convert embedding to tensor
                    input_tensor = torch.tensor(residue_embeddings, dtype=torch.float32).to(self.device)
                    
                    # Run prediction
                    output = model(input_tensor)
                    
                    # Convert to numpy
                    prediction = output.cpu().numpy()
                    
                    # Cache result
                    self._save_to_cache(sequence, property_name, prediction)
                    
                    return prediction
            except Exception as e:
                logger.error(f"Error in model prediction for {property_name}: {e}")
                return self._simplified_prediction(sequence, property_name)
        else:
            return self._simplified_prediction(sequence, property_name)
    
    def _simplified_prediction(self, sequence: str, property_name: str) -> np.ndarray:
        """
        Fallback method to generate simple property predictions based on heuristics.
        Used when the main model is unavailable.
        """
        logger.info(f"Using simplified prediction for {property_name}")
        
        # Length of the sequence
        seq_length = len(sequence)
        
        # Property-specific heuristics
        if property_name == "hydrophobicity":
            # Kyte-Doolittle hydrophobicity scale
            hydrophobicity_scale = {
                'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
            }
            return np.array([hydrophobicity_scale.get(aa, 0.0) for aa in sequence])
            
        elif property_name == "charge":
            # Charge at neutral pH
            charge_scale = {
                'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5
            }
            return np.array([charge_scale.get(aa, 0.0) for aa in sequence])
            
        elif property_name == "secondary_structure":
            # Simplified propensity to form secondary structures (3 classes)
            # Returns values between 0-2: 0=helix, 1=sheet, 2=coil
            ss_propensity = {
                'A': 0, 'E': 0, 'L': 0, 'M': 0, 'Q': 0,  # Helix formers
                'V': 1, 'I': 1, 'F': 1, 'Y': 1, 'W': 1,  # Sheet formers
                'G': 2, 'P': 2, 'S': 2, 'N': 2, 'D': 2   # Coil formers
            }
            # For amino acids not in the dictionary, assign randomly with higher coil probability
            default_probs = [0.3, 0.3, 0.4]  # Helix, sheet, coil
            
            result = np.zeros(seq_length)
            for i, aa in enumerate(sequence):
                if aa in ss_propensity:
                    result[i] = ss_propensity[aa]
                else:
                    result[i] = np.random.choice([0, 1, 2], p=default_probs)
            return result
            
        elif property_name == "solvent_accessibility":
            # Simplified relative solvent accessibility prediction (0-1)
            # Based on simple residue type heuristics
            hydrophobic = {'A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'}
            result = np.zeros(seq_length)
            
            for i, aa in enumerate(sequence):
                # Hydrophobic residues tend to be buried
                if aa in hydrophobic:
                    result[i] = 0.2 + 0.2 * np.random.random()
                else:
                    result[i] = 0.6 + 0.4 * np.random.random()
            
            # Smooth the predictions with a simple window average
            window = 5
            smoothed = np.zeros_like(result)
            for i in range(seq_length):
                start = max(0, i - window // 2)
                end = min(seq_length, i + window // 2 + 1)
                smoothed[i] = np.mean(result[start:end])
            
            return smoothed
            
        elif property_name == "disorder":
            # Simplified disorder prediction (0-1)
            # Higher values indicate higher probability of disorder
            order_promoting = {'W', 'F', 'Y', 'I', 'M', 'L', 'V', 'C'}
            disorder_promoting = {'K', 'E', 'P', 'S', 'Q', 'N', 'D', 'G'}
            
            result = np.zeros(seq_length)
            for i, aa in enumerate(sequence):
                if aa in order_promoting:
                    result[i] = 0.1 + 0.2 * np.random.random()
                elif aa in disorder_promoting:
                    result[i] = 0.7 + 0.3 * np.random.random()
                else:
                    result[i] = 0.4 + 0.2 * np.random.random()
            
            # Smooth the predictions
            window = 9  # Longer window for disorder
            smoothed = np.zeros_like(result)
            for i in range(seq_length):
                start = max(0, i - window // 2)
                end = min(seq_length, i + window // 2 + 1)
                smoothed[i] = np.mean(result[start:end])
            
            return smoothed
            
        elif property_name == "binding_sites":
            # Simplified binding site prediction (0-1)
            # Higher values indicate higher probability of being in a binding site
            binding_prone = {'R', 'K', 'H', 'D', 'E', 'Y', 'W'}
            
            result = np.zeros(seq_length)
            for i, aa in enumerate(sequence):
                if aa in binding_prone:
                    result[i] = 0.6 + 0.4 * np.random.random()
                else:
                    result[i] = 0.1 + 0.2 * np.random.random()
            
            # Create some "patches" of binding sites
            num_patches = seq_length // 30 + 1
            for _ in range(num_patches):
                center = np.random.randint(0, seq_length)
                width = np.random.randint(3, 8)
                start = max(0, center - width // 2)
                end = min(seq_length, center + width // 2 + 1)
                result[start:end] = 0.7 + 0.3 * np.random.random(end - start)
            
            return result
            
        elif property_name == "flexibility":
            # Simplified B-factor-like flexibility prediction
            rigid_aas = {'P', 'W', 'Y', 'F', 'C', 'I', 'L', 'V', 'M'}
            flexible_aas = {'G', 'S', 'D', 'N', 'E', 'K', 'R'}
            
            result = np.zeros(seq_length)
            for i, aa in enumerate(sequence):
                if aa in rigid_aas:
                    result[i] = 0.2 + 0.2 * np.random.random()
                elif aa in flexible_aas:
                    result[i] = 0.7 + 0.3 * np.random.random()
                else:
                    result[i] = 0.4 + 0.3 * np.random.random()
            
            # Smooth and add positional tendencies (termini are more flexible)
            window = 7
            smoothed = np.zeros_like(result)
            for i in range(seq_length):
                start = max(0, i - window // 2)
                end = min(seq_length, i + window // 2 + 1)
                smoothed[i] = np.mean(result[start:end])
            
            # Increase flexibility at termini
            terminus_effect = 10  # Number of residues affected at each end
            for i in range(terminus_effect):
                factor = 1.0 + 0.5 * (1.0 - i / terminus_effect)
                if i < seq_length:
                    smoothed[i] *= factor
                if seq_length - i - 1 >= 0:
                    smoothed[seq_length - i - 1] *= factor
            
            return smoothed
        
        else:
            # Default random output for unknown properties
            return np.random.random(seq_length)
    
    def predict_all_properties(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Predict all available properties for a sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Dictionary mapping property names to predictions
        """
        # Process sequence
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Predict all properties
        results = {}
        for property_name in AVAILABLE_PROPERTIES:
            try:
                results[property_name] = self.predict_property(sequence, property_name)
            except Exception as e:
                logger.error(f"Error predicting {property_name}: {e}")
                # Return empty array for failed predictions
                results[property_name] = np.zeros(len(sequence))
        
        return results


# Singleton instance
_predictor_instance = None

def get_property_predictor() -> ProteinPropertyPredictor:
    """Get or create the property predictor singleton."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ProteinPropertyPredictor()
    return _predictor_instance