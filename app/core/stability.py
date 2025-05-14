import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import pickle
import hashlib

from app.config import settings
from app.services.sequence_utils import validate_sequence, process_sequence
from app.core.embedding import get_embedding_model

# Configure logging
logger = logging.getLogger(__name__)

class ProteinStabilityEstimator:
    """
    Predicts protein stability metrics under various conditions.
    Uses pretrained models to predict melting temperature, free energy, and other stability metrics.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize stability estimation models."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.model_path = model_path or settings.STABILITY_MODEL_PATH
        self.embedding_model = get_embedding_model()
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load the pretrained stability prediction models."""
        try:
            # Load models for different stability metrics
            model_types = ["melting_temp", "free_energy", "ph_stability"]
            
            for model_type in model_types:
                model_file = self.model_path / f"{model_type}_model.pt"
                
                if model_file.exists():
                    logger.info(f"Loading {model_type} stability model")
                    self.models[model_type] = torch.load(model_file, map_location=self.device)
                    self.models[model_type].eval()
                else:
                    logger.warning(f"Model file for {model_type} not found at {model_file}")
                    self.models[model_type] = None
            
            logger.info(f"Loaded {len(self.models)} stability prediction models")
        except Exception as e:
            logger.error(f"Failed to load stability prediction models: {e}")
    
    def _sequence_hash(self, sequence: str, condition_key: str) -> str:
        """Generate a hash for the sequence and conditions for caching purposes."""
        return hashlib.md5(f"{sequence}_{condition_key}".encode()).hexdigest()
    
    def _check_cache(self, sequence: str, condition_key: str) -> Optional[Dict]:
        """Check if stability prediction exists in cache."""
        if not settings.CACHE_ENABLED:
            return None
            
        sequence_hash = self._sequence_hash(sequence, condition_key)
        cache_file = settings.CACHE_DIR / f"stab_{sequence_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached stability prediction: {e}")
                return None
        return None
    
    def _save_to_cache(self, sequence: str, condition_key: str, prediction: Dict):
        """Save stability prediction to cache."""
        if not settings.CACHE_ENABLED:
            return
            
        sequence_hash = self._sequence_hash(sequence, condition_key)
        cache_file = settings.CACHE_DIR / f"stab_{sequence_hash}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(prediction, f)
        except Exception as e:
            logger.warning(f"Failed to cache stability prediction: {e}")
    
    def predict_stability(self, sequence: str, conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict protein stability under specified conditions.
        
        Args:
            sequence: Amino acid sequence string
            conditions: Dictionary of environmental conditions:
                - temperature: Temperature in Celsius (default: 25)
                - ph: pH value (default: 7.0)
                - ionic_strength: Ionic strength in mM (default: 150)
                - pressure: Pressure in atm (default: 1.0)
                - denaturant_conc: Denaturant concentration in M (default: 0.0)
                
        Returns:
            Dictionary containing stability predictions:
                - melting_temp: Predicted melting temperature in Celsius
                - free_energy: Predicted folding free energy in kcal/mol
                - ph_stability: pH range where protein is stable
                - confidence: Confidence score of the prediction (0-1)
        """
        # Validate sequence
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Default conditions
        default_conditions = {
            "temperature": 25.0,
            "ph": 7.0,
            "ionic_strength": 150.0,
            "pressure": 1.0,
            "denaturant_conc": 0.0
        }
        
        # Use default conditions if none provided
        if conditions is None:
            conditions = default_conditions
        else:
            # Fill in missing conditions with defaults
            for key, value in default_conditions.items():
                if key not in conditions:
                    conditions[key] = value
        
        # Create condition key for caching
        condition_key = "_".join([f"{k}_{v}" for k, v in sorted(conditions.items())])
        
        # Check cache
        cached_prediction = self._check_cache(sequence, condition_key)
        if cached_prediction is not None:
            return cached_prediction
        
        # Get sequence embedding
        embedding = self.embedding_model.get_embedding(sequence)
        sequence_embedding = embedding['sequence']  # Use global sequence embedding
        
        # Predict stability using loaded models if available, otherwise use simplified prediction
        if any(model is not None for model in self.models.values()):
            try:
                result = {}
                
                # Convert embedding and conditions to tensor inputs
                embedding_tensor = torch.tensor(sequence_embedding, dtype=torch.float32).to(self.device)
                
                # Create input for models by concatenating embedding with conditions
                condition_values = [conditions[k] for k in sorted(conditions.keys())]
                condition_tensor = torch.tensor(condition_values, dtype=torch.float32).to(self.device)
                model_input = torch.cat([embedding_tensor, condition_tensor]).unsqueeze(0)
                
                # Predict melting temperature
                if self.models.get("melting_temp") is not None:
                    with torch.no_grad():
                        melting_temp = self.models["melting_temp"](model_input).item()
                        result["melting_temp"] = melting_temp
                else:
                    result["melting_temp"] = self._simplified_melting_temp(sequence, conditions)
                
                # Predict free energy
                if self.models.get("free_energy") is not None:
                    with torch.no_grad():
                        free_energy = self.models["free_energy"](model_input).item()
                        result["free_energy"] = free_energy
                else:
                    result["free_energy"] = self._simplified_free_energy(sequence, conditions)
                
                # Predict pH stability range
                if self.models.get("ph_stability") is not None:
                    with torch.no_grad():
                        ph_stability = self.models["ph_stability"](model_input).cpu().numpy()
                        result["ph_stability"] = {
                            "min_ph": float(ph_stability[0][0]),
                            "max_ph": float(ph_stability[0][1])
                        }
                else:
                    result["ph_stability"] = self._simplified_ph_stability(sequence)
                
                # Set confidence based on sequence length
                if len(sequence) < 50 or len(sequence) > 1000:
                    result["confidence"] = 0.5  # Less confident for very short or very long sequences
                else:
                    result["confidence"] = 0.8
                
                # Cache result
                self._save_to_cache(sequence, condition_key, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in stability prediction: {e}")
                # Fallback to simplified prediction
                return self._simplified_stability_prediction(sequence, conditions)
        else:
            return self._simplified_stability_prediction(sequence, conditions)
    
    def _simplified_stability_prediction(self, sequence: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback method using simple heuristics for stability prediction."""
        logger.info("Using simplified stability prediction")
        
        result = {
            "melting_temp": self._simplified_melting_temp(sequence, conditions),
            "free_energy": self._simplified_free_energy(sequence, conditions),
            "ph_stability": self._simplified_ph_stability(sequence),
            "confidence": 0.5  # Lower confidence for simplified prediction
        }
        
        return result
    
    def _simplified_melting_temp(self, sequence: str, conditions: Dict[str, Any]) -> float:
        """Simplified prediction of melting temperature based on sequence properties."""
        # Basic heuristics for melting temperature
        seq_length = len(sequence)
        
        # Count amino acids that influence stability
        stability_increasing = {'C', 'I', 'V', 'W', 'F', 'L', 'M'}
        stability_decreasing = {'G', 'A', 'S', 'P', 'D', 'N'}
        
        increasing_count = sum(1 for aa in sequence if aa in stability_increasing)
        decreasing_count = sum(1 for aa in sequence if aa in stability_decreasing)
        
        # Calculate base melting temperature
        base_tm = 65.0  # Average protein melting temperature
        
        # Adjust for amino acid composition
        stability_score = (increasing_count - decreasing_count) / seq_length
        tm_adjustment = stability_score * 20.0  # Scale factor
        
        # Adjust for protein size (larger proteins often more stable)
        size_factor = min(1.0, seq_length / 300.0) * 5.0
        
        # Adjust for pH (stability typically decreases away from neutral pH)
        ph = conditions.get("ph", 7.0)
        ph_factor = -3.0 * abs(ph - 7.0)
        
        # Adjust for ionic strength (typically stabilizes proteins up to a point)
        ionic_strength = conditions.get("ionic_strength", 150.0)
        ionic_factor = min(5.0, ionic_strength / 150.0 * 5.0)
        
        # Adjust for denaturant
        denaturant = conditions.get("denaturant_conc", 0.0)
        denaturant_factor = -20.0 * denaturant  # Strong negative effect
        
        # Calculate final Tm with some randomness
        melting_temp = base_tm + tm_adjustment + size_factor + ph_factor + ionic_factor + denaturant_factor
        melting_temp += np.random.normal(0, 3.0)  # Add some noise
        
        # Ensure reasonable range
        melting_temp = max(20.0, min(120.0, melting_temp))
        
        return float(melting_temp)
    
    def _simplified_free_energy(self, sequence: str, conditions: Dict[str, Any]) -> float:
        """Simplified prediction of folding free energy."""
        # Basic heuristics for folding free energy
        seq_length = len(sequence)
        
        # More negative value = more stable protein
        base_energy = -7.0  # Average folding energy in kcal/mol
        
        # Count stabilizing residues
        hydrophobic = {'A', 'I', 'L', 'M', 'F', 'V', 'P', 'G'}
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic)
        hydrophobic_contribution = -0.05 * hydrophobic_count
        
        # Size contribution (larger proteins have more interactions)
        size_contribution = -0.01 * seq_length
        
        # Temperature effect (higher temp = less stable)
        temperature = conditions.get("temperature", 25.0)
        temp_contribution = 0.1 * (temperature - 25.0) / 10.0
        
        # Denaturant effect
        denaturant = conditions.get("denaturant_conc", 0.0)
        denaturant_contribution = 5.0 * denaturant
        
        # Calculate final folding energy with some randomness
        free_energy = base_energy + hydrophobic_contribution + size_contribution + temp_contribution + denaturant_contribution
        free_energy += np.random.normal(0, 1.0)  # Add some noise
        
        # Ensure reasonable range
        free_energy = max(-25.0, min(5.0, free_energy))
        
        return float(free_energy)
    
    def _simplified_ph_stability(self, sequence: str) -> Dict[str, float]:
        """Simplified prediction of pH stability range."""
        # Count charged residues
        acidic = {'D', 'E'}
        basic = {'K', 'R', 'H'}
        
        acidic_count = sum(1 for aa in sequence if aa in acidic)
        basic_count = sum(1 for aa in sequence if aa in basic)
        
        # Calculate relative proportions
        seq_length = len(sequence)
        acidic_fraction = acidic_count / seq_length
        basic_fraction = basic_count / seq_length
        
        # Base pH range (most proteins are stable around neutral pH)
        min_ph = 5.0
        max_ph = 9.0
        
        # Adjust based on charged residues
        if acidic_fraction > 0.12:  # High acidic content
            min_ph -= 1.0
            max_ph -= 0.5
        
        if basic_fraction > 0.12:  # High basic content
            min_ph += 0.5
            max_ph += 1.0
        
        # Add some noise
        min_ph += np.random.normal(0, 0.3)
        max_ph += np.random.normal(0, 0.3)
        
        # Ensure reasonable range and min < max
        min_ph = max(2.0, min(6.5, min_ph))
        max_ph = max(8.0, min(12.0, max_ph))
        max_ph = max(max_ph, min_ph + 1.5)  # Ensure reasonable gap
        
        return {
            "min_ph": float(min_ph),
            "max_ph": float(max_ph)
        }


# Singleton instance
_stability_estimator_instance = None

def get_stability_estimator() -> ProteinStabilityEstimator:
    """Get or create the stability estimator singleton."""
    global _stability_estimator_instance
    if _stability_estimator_instance is None:
        _stability_estimator_instance = ProteinStabilityEstimator()
    return _stability_estimator_instance