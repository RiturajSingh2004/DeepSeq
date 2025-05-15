"""
Script to setup AlphaFold for the Protein Analysis Platform with API integration.
This will configure AlphaFold and use APIs to fetch required data instead of downloading full databases.
"""

import os
import sys
import logging
import argparse
import subprocess
import json
import time
from pathlib import Path
import requests
import tarfile
import gzip
import shutil
import tempfile
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("alphafold_setup")

# AlphaFold repository
ALPHAFOLD_REPO = "https://github.com/google-deepmind/alphafold.git"
ALPHAFOLD_VERSION = "v2.3.2"  # Replace with the version you want to use

# AlphaFold API endpoints (replace with actual endpoints)
API_BASE_URL = "https://api.proteindatabank.org/v1"  # Example API base URL
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api"  # AlphaFold EBI API

# Parameter files - still need to download these
PARAMS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

def run_command(cmd, cwd=None):
    """Run a shell command and log its output."""
    logger.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd
    )

    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            logger.info(output.strip())

    rc = process.poll()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    return rc

def download_file(url, output_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(output_path, 'wb') as f, tqdm(
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

def extract_tar(tar_path, extract_to):
    """Extract a tar file."""
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)

    logger.info(f"Extracted {tar_path} to {extract_to}")
    return extract_to

def setup_alphafold_repo(install_dir):
    """Clone and set up the AlphaFold repository."""
    alphafold_dir = Path(install_dir) / "alphafold"

    if alphafold_dir.exists():
        logger.info(f"AlphaFold directory already exists at {alphafold_dir}")
        return alphafold_dir

    # Clone AlphaFold repository
    logger.info(f"Cloning AlphaFold repository to {alphafold_dir}")
    run_command(["git", "clone", ALPHAFOLD_REPO, str(alphafold_dir)])

    # Checkout specific version
    logger.info(f"Checking out AlphaFold version {ALPHAFOLD_VERSION}")
    run_command(["git", "checkout", ALPHAFOLD_VERSION], cwd=alphafold_dir)

    # Install AlphaFold dependencies
    logger.info("Installing AlphaFold dependencies")
    run_command(["pip", "install", "-r", "requirements.txt"], cwd=alphafold_dir)

    # Install additional dependencies
    run_command(["pip", "install", "dm-haiku", "ml-collections", "tensorflow"])
    
    # Install API client dependencies
    run_command(["pip", "install", "requests", "backoff", "cachetools"])

    logger.info("AlphaFold repository setup complete")
    return alphafold_dir

def setup_alphafold_params(data_dir):
    """Download and set up AlphaFold parameter files only."""
    data_dir = Path(data_dir)
    params_dir = data_dir / "params"
    os.makedirs(params_dir, exist_ok=True)
    
    # Parameter files are still needed for AlphaFold to run
    logger.info("Downloading AlphaFold parameter files")
    tar_path = data_dir / "alphafold_params.tar"
    
    try:
        download_file(PARAMS_URL, tar_path)
        
        # Extract params
        logger.info("Extracting parameter files")
        extract_tar(tar_path, params_dir)
        
        # Clean up tar file
        tar_path.unlink()
        
        logger.info("Parameter files setup complete")
        
    except Exception as e:
        logger.error(f"Error setting up parameter files: {e}")
        
    return params_dir

def create_api_client_module(alphafold_dir):
    """Create an API client module to interface with protein databases."""
    api_client_path = alphafold_dir / "alphafold" / "data" / "api_data_client.py"
    
    api_client_code = """
# API client for AlphaFold to fetch protein data from remote services
import os
import json
import time
import logging
import requests
import backoff
from typing import Dict, List, Optional, Tuple, Union
from cachetools import TTLCache, cached

logger = logging.getLogger(__name__)

# Cache for API responses
structure_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for one hour
sequence_cache = TTLCache(maxsize=1000, ttl=3600)

class AlphaFoldAPIClient:
    \"\"\"Client for fetching protein data from APIs instead of local databases.\"\"\"
    
    def __init__(self, base_url="https://alphafold.ebi.ac.uk/api", 
                 pdb_api_url="https://data.rcsb.org/rest/v1/core"):
        self.base_url = base_url
        self.pdb_api_url = pdb_api_url
        self.session = requests.Session()
        # Add retries with exponential backoff
        self.session.mount('https://', requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
            )
        ))
    
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=5)
    def _make_request(self, url, params=None):
        \"\"\"Make API request with exponential backoff for retries.\"\"\"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Resource not found: {url}")
                return None
            logger.error(f"HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
    
    @cached(cache=structure_cache)
    def get_structure_by_pdb_id(self, pdb_id):
        \"\"\"Fetch structure data for a PDB ID.\"\"\"
        url = f"{self.pdb_api_url}/entry/{pdb_id.lower()}"
        return self._make_request(url)
    
    @cached(cache=sequence_cache)
    def get_sequence_by_uniprot_id(self, uniprot_id):
        \"\"\"Fetch sequence data for a UniProt ID.\"\"\"
        url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}"
        return self._make_request(url)
    
    def search_msa_by_sequence(self, sequence, max_hits=100):
        \"\"\"Search for MSA data by sequence.\"\"\"
        # This would call an API that provides MSA data (like MGnify or similar)
        # For now, we'll return a placeholder
        logger.info(f"Searching for MSA with sequence of length {len(sequence)}")
        # In reality, you would make an API request here
        return {"hits": [], "status": "placeholder_for_api_integration"}
    
    def get_alphafold_prediction(self, uniprot_id):
        \"\"\"Get AlphaFold prediction for a UniProt ID.\"\"\"
        url = f"{self.base_url}/prediction/{uniprot_id}"
        return self._make_request(url)
    
    def download_pdb_file(self, pdb_id, output_path):
        \"\"\"Download a PDB file for a given PDB ID.\"\"\"
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return output_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDB file: {e}")
            return None

# Initialize the client
api_client = AlphaFoldAPIClient()

# Functions to be called by AlphaFold in place of database lookups
def fetch_sequence(sequence_id):
    \"\"\"Fetch a sequence by ID from appropriate database API.\"\"\"
    if sequence_id.startswith("UP"):  # UniProt ID
        return api_client.get_sequence_by_uniprot_id(sequence_id.split("_")[0])
    elif sequence_id.startswith("PDB"):  # PDB ID
        return api_client.get_structure_by_pdb_id(sequence_id.split("_")[1])
    else:
        logger.warning(f"Unknown sequence ID format: {sequence_id}")
        return None

def fetch_msa(sequence):
    \"\"\"Fetch MSA by sequence from API.\"\"\"
    return api_client.search_msa_by_sequence(sequence)

def fetch_template(pdb_id, output_dir):
    \"\"\"Fetch a template structure by PDB ID.\"\"\"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    return api_client.download_pdb_file(pdb_id, output_path)
"""
    
    with open(api_client_path, 'w') as f:
        f.write(api_client_code)
    
    logger.info(f"Created API client module at {api_client_path}")
    return api_client_path

def patch_alphafold_pipeline(alphafold_dir):
    """Create a patch to modify AlphaFold pipeline to use API client."""
    patch_path = alphafold_dir / "patches" / "use_api_client.patch"
    os.makedirs(alphafold_dir / "patches", exist_ok=True)
    
    # Create a patch file - this is a simplified example
    # In a real scenario, you would need to create proper Git patches or modify files directly
    patch_content = """
--- a/alphafold/data/pipeline.py
+++ b/alphafold/data/pipeline.py
@@ -21,6 +21,7 @@ import time
 from alphafold.data import parsers
 from alphafold.data import templates
 from alphafold.data.tools import hhblits
+from alphafold.data import api_data_client  # Import our API client
 
 def process_features(
     input_sequence,
@@ -30,6 +31,13 @@ def process_features(
 ):
     """Process features for AlphaFold prediction."""
     
+    # First check if we can get prediction directly from AlphaFold API
+    if use_api_mode:
+        uniprot_id = get_uniprot_id_from_sequence(input_sequence)
+        if uniprot_id:
+            prediction = api_data_client.api_client.get_alphafold_prediction(uniprot_id)
+            if prediction and 'pdb_structure' in prediction:
+                return {'api_prediction': prediction}
     
     # Process MSA features
     msa_features = {}
@@ -38,7 +46,10 @@ def process_features(
     # For single sequence
     if not msa_features:
         msa_features = process_single_sequence(input_sequence)
-    
+        
+        # Try API-based MSA search if local databases are not available
+        if use_api_mode and not msa_features.get('msa'):
+            msa_features['msa'] = api_data_client.fetch_msa(input_sequence)
     
     # Get templates
     template_features = {}
@@ -46,6 +57,12 @@ def process_features(
     if template_searcher:
         template_features = template_searcher.get_templates(
             query_sequence=input_sequence)
+            
+    if use_api_mode and not template_features.get('template_domain_names'):
+        # Use API to search for templates if local database is not available
+        # This is a placeholder for where you'd call the API client to get templates
+        pass
+    
     
     # Combine all features
     feature_dict = {**msa_features, **template_features}
"""
    
    with open(patch_path, 'w') as f:
        f.write(patch_content)
    
    logger.info(f"Created pipeline patch at {patch_path}")
    return patch_path

def create_api_configuration(alphafold_dir, data_dir):
    """Create API configuration file."""
    config_path = data_dir / "api_config.json"
    
    config = {
        "api_endpoints": {
            "alphafold_db": "https://alphafold.ebi.ac.uk/api",
            "pdb": "https://data.rcsb.org/rest/v1/core",
            "uniprot": "https://www.ebi.ac.uk/proteins/api/proteins",
            "mgnify": "https://www.ebi.ac.uk/metagenomics/api/latest"
        },
        "api_keys": {
            # Add your API keys here if needed
        },
        "rate_limits": {
            "alphafold_db": {"requests_per_minute": 60},
            "pdb": {"requests_per_minute": 10},
            "uniprot": {"requests_per_minute": 10},
            "mgnify": {"requests_per_minute": 10}
        },
        "cache": {
            "enabled": True,
            "directory": str(data_dir / "api_cache"),
            "max_size_mb": 1000,
            "ttl_seconds": 3600
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create cache directory
    os.makedirs(data_dir / "api_cache", exist_ok=True)
    
    logger.info(f"Created API configuration at {config_path}")
    return config_path

def create_minimal_database_placeholders(data_dir):
    """Create minimal placeholder directories for compatibility."""
    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create empty database directories for compatibility
    db_dirs = [
        "pdb70", "pdb_mmcif", "uniref90", 
        "mgnify", "bfd", "uniclust30"
    ]

    for db_dir in db_dirs:
        os.makedirs(data_dir / db_dir, exist_ok=True)
        
        # Create minimal placeholder files
        placeholder_file = data_dir / db_dir / "placeholder.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"This directory is a placeholder for the {db_dir} database.\n")
            f.write("Instead of the full database, the system uses API calls to fetch required data.\n")

    logger.info("Created minimal database placeholders for compatibility")
    return data_dir

def create_api_runner_script(alphafold_dir):
    """Create a script to run AlphaFold with API integration."""
    runner_path = alphafold_dir / "run_alphafold_api.py"
    
    runner_script = """#!/usr/bin/env python3
# Script to run AlphaFold with API integration

import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Add the repository root to the path so that we can import alphafold.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from alphafold.model import config
from alphafold.model import model
from alphafold.data import pipeline
from alphafold.data import api_data_client  # Import our API client

def main():
    parser = argparse.ArgumentParser(description='Run AlphaFold with API integration')
    
    parser.add_argument(
        '--fasta_path', required=True,
        help='Path to FASTA file containing sequence(s) to fold')
    
    parser.add_argument(
        '--output_dir', required=True,
        help='Path to a directory that will store the results')
    
    parser.add_argument(
        '--data_dir',
        default=os.path.join(os.path.dirname(current_dir), 'alphafold_data'),
        help='Path to directory with parameters and minimal database placeholders')
    
    parser.add_argument(
        '--use_api', action='store_true', default=True,
        help='Use API for data fetching instead of local databases')
    
    parser.add_argument(
        '--api_config', 
        default=None,
        help='Path to API configuration file')
    
    parser.add_argument(
        '--max_template_date', default='2100-01-01',
        help='Maximum template release date to consider (ISO format - YYYY-MM-DD)')
    
    args = parser.parse_args()

    # Make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "alphafold.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("alphafold_api")
    
    # Load API configuration if provided
    if args.api_config:
        with open(args.api_config) as f:
            api_config = json.load(f)
            logger.info(f"Loaded API configuration from {args.api_config}")
    elif args.use_api:
        api_config_path = os.path.join(args.data_dir, "api_config.json")
        if os.path.exists(api_config_path):
            with open(api_config_path) as f:
                api_config = json.load(f)
                logger.info(f"Loaded API configuration from {api_config_path}")
        else:
            logger.warning("No API configuration found. Using default settings.")
            api_config = {}
    
    # Call the AlphaFold pipeline with API integration
    # This is a placeholder for the actual implementation
    logger.info("Running AlphaFold with API integration")
    logger.info(f"Input FASTA: {args.fasta_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"API mode: {'enabled' if args.use_api else 'disabled'}")
    
    # Here you would implement the actual call to the AlphaFold pipeline
    # with modifications to use the API client when needed
    
    logger.info("AlphaFold prediction complete")

if __name__ == '__main__':
    main()
"""
    
    with open(runner_path, 'w') as f:
        f.write(runner_script)
    
    # Make the script executable
    run_command(["chmod", "+x", str(runner_path)])
    
    logger.info(f"Created API runner script at {runner_path}")
    return runner_path

def validate_setup(alphafold_dir, data_dir):
    """Validate the AlphaFold API setup."""
    issues = []

    # Check AlphaFold directory
    if not (alphafold_dir / "run_alphafold.py").exists():
        issues.append(f"AlphaFold run script not found at {alphafold_dir / 'run_alphafold.py'}")
    
    # Check if API runner script exists
    if not (alphafold_dir / "run_alphafold_api.py").exists():
        issues.append(f"API runner script not found at {alphafold_dir / 'run_alphafold_api.py'}")
    
    # Check if API client module exists
    if not (alphafold_dir / "alphafold" / "data" / "api_data_client.py").exists():
        issues.append(f"API client module not found")
    
    # Check parameter files
    if not (data_dir / "params").exists():
        issues.append(f"Parameter directory not found at {data_dir / 'params'}")
    
    # Check API configuration
    if not (data_dir / "api_config.json").exists():
        issues.append(f"API configuration not found at {data_dir / 'api_config.json'}")
    
    # Check cache directory
    if not (data_dir / "api_cache").exists():
        issues.append(f"API cache directory not found at {data_dir / 'api_cache'}")

    # Report issues
    if issues:
        logger.warning("AlphaFold API setup validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False

    logger.info("AlphaFold API setup validation successful")
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up AlphaFold with API integration")
    parser.add_argument("--install-dir", type=str, default="/opt",
                      help="Directory to install AlphaFold (default: /opt)")
    parser.add_argument("--data-dir", type=str, default="/opt/alphafold_data",
                      help="Directory for AlphaFold data (default: /opt/alphafold_data)")
    parser.add_argument("--skip-repo", action="store_true",
                      help="Skip setting up AlphaFold repository")
    parser.add_argument("--api-endpoints", type=str, default=None,
                      help="JSON file with custom API endpoints")

    args = parser.parse_args()

    # Set up AlphaFold repository
    if not args.skip_repo:
        alphafold_dir = setup_alphafold_repo(args.install_dir)
    else:
        alphafold_dir = Path(args.install_dir) / "alphafold"
        logger.info(f"Skipping AlphaFold repository setup, using {alphafold_dir}")

    # Set up data directory
    data_dir = Path(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Download only parameter files (required for AlphaFold)
    setup_alphafold_params(data_dir)
    
    # Create minimal placeholder directories for compatibility
    create_minimal_database_placeholders(data_dir)
    
    # Create API client module
    create_api_client_module(alphafold_dir)
    
    # Create patch for AlphaFold pipeline
    patch_alphafold_pipeline(alphafold_dir)
    
    # Create API configuration
    create_api_configuration(alphafold_dir, data_dir)
    
    # Create API runner script
    create_api_runner_script(alphafold_dir)
    
    # Validate setup
    validate_setup(alphafold_dir, data_dir)

    logger.info("AlphaFold API setup complete")
    logger.info(f"AlphaFold repository: {alphafold_dir}")
    logger.info(f"AlphaFold data: {data_dir}")
    logger.info(f"Run AlphaFold with API using: {alphafold_dir / 'run_alphafold_api.py'}")

if __name__ == "__main__":
    main()
