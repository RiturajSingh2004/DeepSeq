#!/usr/bin/env python3
"""
Script to download pretrained models for the Protein Analysis Platform.
This downloads and sets up the necessary model files for embedding, property prediction,
classification, and stability estimation.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import requests
import tqdm
import zipfile
import hashlib
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_downloader")

# Model URLs and checksums - these should be updated with actual model URLs
# In a real implementation, these would point to model storage
MODEL_URLS = {
    "embedding": {
        "url": "https://your-model-storage/embedding_model.zip",
        "md5": "placeholder_checksum_for_embedding_model",
    },
    "property": {
        "url": "https://your-model-storage/property_model.zip",
        "md5": "placeholder_checksum_for_property_model",
    },
    "classification": {
        "url": "https://your-model-storage/classification_model.zip", 
        "md5": "placeholder_checksum_for_classification_model",
    },
    "stability": {
        "url": "https://your-model-storage/stability_model.zip",
        "md5": "placeholder_checksum_for_stability_model",
    }
}

# Hugging Face models
HF_MODELS = {
    "esm2": "facebook/esm2_t33_650M_UR50D"
}

def download_file(url, output_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, 'wb') as f, tqdm.tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            size = f.write(chunk)
            bar.update(size)
    
    return output_path

def verify_checksum(file_path, expected_md5):
    """Verify MD5 checksum of downloaded file."""
    if expected_md5 == "placeholder_checksum_for_embedding_model":
        logger.warning(f"Skipping checksum verification for {file_path} (placeholder checksum)")
        return True
        
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    file_md5 = md5_hash.hexdigest()
    if file_md5 != expected_md5:
        logger.error(f"Checksum mismatch for {file_path}")
        logger.error(f"Expected: {expected_md5}")
        logger.error(f"Got: {file_md5}")
        return False
    
    return True

def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extracted {zip_path} to {extract_to}")
    return extract_to

def download_custom_models(model_dir):
    """Download custom trained models."""
    os.makedirs(model_dir, exist_ok=True)
    
    for model_name, info in MODEL_URLS.items():
        model_subdir = model_dir / model_name
        os.makedirs(model_subdir, exist_ok=True)
        
        zip_path = model_dir / f"{model_name}_model.zip"
        
        try:
            # Download model
            logger.info(f"Downloading {model_name} model")
            download_file(info["url"], zip_path)
            
            # Verify checksum
            if verify_checksum(zip_path, info["md5"]):
                # Extract model
                extract_zip(zip_path, model_subdir)
                logger.info(f"Successfully installed {model_name} model")
            else:
                logger.error(f"Checksum verification failed for {model_name} model")
            
            # Clean up zip file
            zip_path.unlink()
            
        except Exception as e:
            logger.error(f"Error downloading {model_name} model: {e}")

def download_huggingface_models(model_dir):
    """Download models from Hugging Face."""
    for model_name, model_id in HF_MODELS.items():
        try:
            logger.info(f"Downloading {model_name} model from Hugging Face: {model_id}")
            
            # Create model directory
            hf_dir = model_dir / model_name
            os.makedirs(hf_dir, exist_ok=True)
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            
            # Save locally
            tokenizer.save_pretrained(hf_dir)
            model.save_pretrained(hf_dir)
            
            logger.info(f"Successfully installed {model_name} model from Hugging Face")
        except Exception as e:
            logger.error(f"Error downloading {model_name} model from Hugging Face: {e}")

def setup_simplified_models(model_dir):
    """
    Create simplified placeholder models when actual models are not available.
    These are simple PyTorch models that output random values but have the
    expected API interfaces.
    """
    from torch import nn
    
    # Create property prediction models
    property_dir = model_dir / "property"
    os.makedirs(property_dir, exist_ok=True)
    
    # List of properties to create simplified models for
    properties = [
        "hydrophobicity", "charge", "secondary_structure", 
        "solvent_accessibility", "disorder", "binding_sites", "flexibility"
    ]
    
    for prop in properties:
        # Create a simple MLP model
        model = nn.Sequential(
            nn.Linear(1024, 256),  # Assuming 1024-dim input embedding
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output value per residue
        )
        
        # Save the model
        torch.save(model, property_dir / f"{prop}_model.pt")
        logger.info(f"Created simplified model for {prop}")
    
    # Create classification models
    classification_dir = model_dir / "classification"
    os.makedirs(classification_dir, exist_ok=True)
    
    # List of classification types
    classification_types = [
        "subcellular_localization", "antimicrobial_potential", 
        "toxicity", "enzyme_class", "gpcr_type", "protein_family"
    ]
    
    # Class counts for each classification type
    class_counts = {
        "subcellular_localization": 10,
        "antimicrobial_potential": 2,
        "toxicity": 2,
        "enzyme_class": 7,
        "gpcr_type": 4,
        "protein_family": 7
    }
    
    for cls_type in classification_types:
        num_classes = class_counts.get(cls_type, 5)
        
        # Create a simple classification model
        model = nn.Sequential(
            nn.Linear(1024, 256),  # Assuming 1024-dim input embedding
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Save the model
        torch.save(model, classification_dir / f"{cls_type}_model.pt")
        logger.info(f"Created simplified model for {cls_type} classification")
    
    # Create stability prediction models
    stability_dir = model_dir / "stability"
    os.makedirs(stability_dir, exist_ok=True)
    
    # Stability prediction models
    stability_types = ["melting_temp", "free_energy", "ph_stability"]
    
    for stab_type in stability_types:
        output_size = 2 if stab_type == "ph_stability" else 1
        
        # Create a simple regression model
        model = nn.Sequential(
            nn.Linear(1029, 256),  # 1024 for embedding + 5 for conditions
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        # Save the model
        torch.save(model, stability_dir / f"{stab_type}_model.pt")
        logger.info(f"Created simplified model for {stab_type} prediction")
    
    logger.info("All simplified models have been created successfully")

def main():
    parser = argparse.ArgumentParser(description="Download pretrained models for Protein Analysis Platform")
    parser.add_argument("--model-dir", type=str, default="./app/models/pretrained",
                        help="Directory to save models (default: ./app/models/pretrained)")
    parser.add_argument("--simplified", action="store_true", 
                        help="Use simplified placeholder models instead of downloading real models")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip downloading models from Hugging Face")
    parser.add_argument("--skip-custom", action="store_true",
                        help="Skip downloading custom models")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Setting up models in {model_dir}")
    
    if args.simplified:
        logger.info("Creating simplified placeholder models")
        setup_simplified_models(model_dir)
    else:
        if not args.skip_hf:
            download_huggingface_models(model_dir)
        
        if not args.skip_custom:
            download_custom_models(model_dir)
    
    logger.info("Model setup complete")

if __name__ == "__main__":
    main()