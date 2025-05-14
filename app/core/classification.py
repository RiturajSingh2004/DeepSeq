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

# Classification types
CLASSIFICATION_TYPES = [
    "subcellular_localization", 
    "antimicrobial_potential",
    "toxicity",
    "enzyme_class",
    "gpcr_type",
    "protein_family"
]

# Classes for each classification type
CLASSIFICATION_CLASSES = {
    "subcellular_localization": [
        "cytoplasm", "nucleus", "mitochondria", "endoplasmic_reticulum", 
        "golgi_apparatus", "peroxisome", "lysosome", "plasma_membrane", 
        "extracellular", "chloroplast"
    ],
    "antimicrobial_potential": [
        "antimicrobial", "non_antimicrobial"
    ],
    "toxicity": [
        "toxic", "non_toxic"
    ],
    "enzyme_class": [
        "oxidoreductase", "transferase", "hydrolase", 
        "lyase", "isomerase", "ligase", "translocase"
    ],
    "gpcr_type": [
        "class_a", "class_b", "class_c", "class_f"
    ],
    "protein_family": [
        "kinase", "phosphatase", "protease", "ion_channel", 
        "transcription_factor", "transporter", "receptor"
    ]
}

class ProteinClassifier:
    """
    Handles classification of protein sequences for various functional properties.
    Uses pretrained models to map from embedding space to classification outputs.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the classifier models."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.USE_GPU else "cpu")
        self.model_path = model_path or settings.CLASSIFICATION_MODEL_PATH
        self.embedding_model = get_embedding_model()
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load the pretrained classification models."""
        try:
            for class_type in CLASSIFICATION_TYPES:
                model_file = self.model_path / f"{class_type}_model.pt"
                
                if model_file.exists():
                    logger.info(f"Loading {class_type} classification model")
                    self.models[class_type] = self._load_specific_model(model_file)
                else:
                    logger.warning(f"Model file for {class_type} not found at {model_file}")
                    # Use simplified model as fallback
                    self.models[class_type] = None
            
            logger.info(f"Loaded {len(self.models)} classification models")
        except Exception as e:
            logger.error(f"Failed to load classification models: {e}")
    
    def _load_specific_model(self, model_path: Path) -> Any:
        """Load a specific classification model."""
        try:
            # Load the PyTorch model
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def _sequence_hash(self, sequence: str, class_type: str) -> str:
        """Generate a hash for caching purposes."""
        return hashlib.md5(f"{sequence}_{class_type}".encode()).hexdigest()
    
    def _check_cache(self, sequence: str, class_type: str) -> Optional[Dict]:
        """Check if classification exists in cache."""
        if not settings.CACHE_ENABLED:
            return None
            
        sequence_hash = self._sequence_hash(sequence, class_type)
        cache_file = settings.CACHE_DIR / f"class_{sequence_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached classification: {e}")
                return None
        return None
    
    def _save_to_cache(self, sequence: str, class_type: str, classification: Dict):
        """Save classification to cache."""
        if not settings.CACHE_ENABLED:
            return
            
        sequence_hash = self._sequence_hash(sequence, class_type)
        cache_file = settings.CACHE_DIR / f"class_{sequence_hash}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(classification, f)
        except Exception as e:
            logger.warning(f"Failed to cache classification: {e}")
    
    def classify(self, sequence: str, class_type: str) -> Dict[str, float]:
        """
        Classify a protein sequence for the given classification type.
        
        Args:
            sequence: Amino acid sequence string
            class_type: Classification type
            
        Returns:
            Dictionary mapping class labels to probabilities
        """
        # Validate inputs
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
            
        if class_type not in CLASSIFICATION_TYPES:
            raise ValueError(f"Unknown classification type: {class_type}. Available types: {CLASSIFICATION_TYPES}")
        
        # Check cache
        cached_classification = self._check_cache(sequence, class_type)
        if cached_classification is not None:
            return cached_classification
        
        # Get sequence embedding
        embedding = self.embedding_model.get_embedding(sequence)
        sequence_embedding = embedding['sequence']  # Use global sequence embedding for classification
        
        # Get appropriate model
        model = self.models.get(class_type)
        
        # Use pretrained model if available, otherwise use simplified classification
        if model is not None:
            try:
                with torch.no_grad():
                    # Convert embedding to tensor
                    input_tensor = torch.tensor(sequence_embedding, dtype=torch.float32).reshape(1, -1).to(self.device)
                    
                    # Run prediction
                    output = model(input_tensor)
                    
                    # Apply softmax to get probabilities
                    probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
                    
                    # Map to class labels
                    class_labels = CLASSIFICATION_CLASSES.get(class_type, [f"class_{i}" for i in range(len(probabilities))])
                    classification = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
                    
                    # Cache result
                    self._save_to_cache(sequence, class_type, classification)
                    
                    return classification
            except Exception as e:
                logger.error(f"Error in model classification for {class_type}: {e}")
                return self._simplified_classification(sequence, class_type)
        else:
            return self._simplified_classification(sequence, class_type)
    
    def _simplified_classification(self, sequence: str, class_type: str) -> Dict[str, float]:
        """
        Fallback method to generate simple classifications based on sequence heuristics.
        Used when the main model is unavailable.
        """
        logger.info(f"Using simplified classification for {class_type}")
        
        # Get class labels for the classification type
        class_labels = CLASSIFICATION_CLASSES.get(class_type, [f"class_{i}" for i in range(5)])
        num_classes = len(class_labels)
        
        # Simple heuristics based on sequence composition
        aa_counts = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            aa_counts[aa] = sequence.count(aa) / len(sequence)  # Normalized counts
        
        # Different heuristics for different classification types
        if class_type == "subcellular_localization":
            # Very simplified localization prediction based on amino acid composition
            probabilities = np.zeros(len(class_labels))
            
            # High K, R -> nucleus
            if aa_counts['K'] + aa_counts['R'] > 0.15:
                nucleus_idx = class_labels.index("nucleus")
                probabilities[nucleus_idx] = 0.6
            
            # High hydrophobic (L, I, V, F, W) -> membrane
            hydrophobic = aa_counts['L'] + aa_counts['I'] + aa_counts['V'] + aa_counts['F'] + aa_counts['W']
            if hydrophobic > 0.35:
                membrane_idx = class_labels.index("plasma_membrane")
                probabilities[membrane_idx] = 0.5
            
            # High C -> extracellular
            if aa_counts['C'] > 0.05:
                extracellular_idx = class_labels.index("extracellular")
                probabilities[extracellular_idx] = 0.4
            
            # Default to cytoplasm if nothing else strong
            if np.max(probabilities) < 0.4:
                cytoplasm_idx = class_labels.index("cytoplasm")
                probabilities[cytoplasm_idx] = 0.4
            
            # Normalize and add randomness
            probabilities = probabilities + 0.1 * np.random.random(len(probabilities))
            probabilities = probabilities / np.sum(probabilities)
            
        elif class_type == "antimicrobial_potential":
            # Simple heuristic: high fraction of K, R, H (positive charge) suggests antimicrobial
            positive_charge = aa_counts['K'] + aa_counts['R'] + aa_counts['H']
            
            if positive_charge > 0.2:
                p_antimicrobial = 0.7 + 0.3 * np.random.random()
            else:
                p_antimicrobial = 0.3 * np.random.random()
                
            probabilities = np.array([p_antimicrobial, 1.0 - p_antimicrobial])
            
        elif class_type == "toxicity":
            # Simple heuristic: high C content and short sequence suggests toxin
            is_short = len(sequence) < 100
            high_cys = aa_counts['C'] > 0.08
            
            if is_short and high_cys:
                p_toxic = 0.7 + 0.3 * np.random.random()
            elif is_short or high_cys:
                p_toxic = 0.4 + 0.3 * np.random.random()
            else:
                p_toxic = 0.3 * np.random.random()
                
            probabilities = np.array([p_toxic, 1.0 - p_toxic])
            
        elif class_type == "enzyme_class":
            # Basic heuristics for enzyme classes
            probabilities = np.random.random(len(class_labels))
            
            # Higher probability for hydrolase if high S, D content
            if aa_counts['S'] + aa_counts['D'] > 0.15:
                hydrolase_idx = class_labels.index("hydrolase")
                probabilities[hydrolase_idx] = 0.4 + 0.3 * np.random.random()
            
            # Higher probability for oxidoreductase if high C content
            if aa_counts['C'] > 0.05:
                oxidoreductase_idx = class_labels.index("oxidoreductase")
                probabilities[oxidoreductase_idx] = 0.4 + 0.3 * np.random.random()
            
            # Normalize
            probabilities = probabilities / np.sum(probabilities)
            
        else:
            # For other classification types, generate random probabilities
            probabilities = np.random.random(num_classes)
            probabilities = probabilities / np.sum(probabilities)
        
        # Create classification dictionary
        classification = {label: float(prob) for label, prob in zip(class_labels, probabilities)}
        
        return classification
    
    def classify_all(self, sequence: str) -> Dict[str, Dict[str, float]]:
        """
        Run all available classifications on a sequence.
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Dictionary mapping classification types to their results
        """
        # Process sequence
        sequence = process_sequence(sequence)
        if not validate_sequence(sequence):
            raise ValueError("Invalid protein sequence")
        
        # Run all classifications
        results = {}
        for class_type in CLASSIFICATION_TYPES:
            try:
                results[class_type] = self.classify(sequence, class_type)
            except Exception as e:
                logger.error(f"Error in classification {class_type}: {e}")
                # Return empty result for failed classifications
                results[class_type] = {label: 0.0 for label in CLASSIFICATION_CLASSES.get(class_type, [])}
        
        return results


# Singleton instance
_classifier_instance = None

def get_classifier() -> ProteinClassifier:
    """Get or create the classifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ProteinClassifier()
    return _classifier_instance