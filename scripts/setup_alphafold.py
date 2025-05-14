#!/usr/bin/env python3
"""
Script to setup AlphaFold for the Protein Analysis Platform.
This will download and configure AlphaFold and its required databases.
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import requests
import tarfile
import gzip
import shutil
import tempfile
import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("alphafold_setup")

# AlphaFold repository
ALPHAFOLD_REPO = "https://github.com/google-deepmind/alphafold.git"
ALPHAFOLD_VERSION = "v2.3.2"  # Replace with the version you want to use

# Database URLs - These would be replaced with actual database URLs
# Note: These databases are large and require significant disk space
DATABASE_URLS = {
    "params": "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
    "reduced_dbs": "https://storage.googleapis.com/alphafold/reduced_dbs_20221201.tar",
    # Add other database URLs as needed
}

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
    
    logger.info("AlphaFold repository setup complete")
    return alphafold_dir

def download_databases(data_dir):
    """Download and set up AlphaFold databases."""
    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and extract each database
    for db_name, url in DATABASE_URLS.items():
        db_dir = data_dir / db_name
        
        if db_dir.exists() and db_name != "params":  # Always update params
            logger.info(f"Database {db_name} already exists at {db_dir}")
            continue
        
        os.makedirs(db_dir, exist_ok=True)
        
        # Download database
        logger.info(f"Downloading {db_name} database")
        tar_path = data_dir / f"{db_name}.tar"
        
        try:
            download_file(url, tar_path)
            
            # Extract database
            logger.info(f"Extracting {db_name} database")
            extract_tar(tar_path, db_dir)
            
            # Clean up tar file
            tar_path.unlink()
            
            logger.info(f"Database {db_name} setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up {db_name} database: {e}")
    
    logger.info("Database setup complete")
    return data_dir

def setup_minimal_databases(data_dir):
    """
    Create minimal placeholder databases for testing.
    This is used when the full databases are not needed or available.
    """
    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create minimal parameter files
    params_dir = data_dir / "params"
    os.makedirs(params_dir, exist_ok=True)
    
    # Create empty database directories
    db_dirs = [
        "pdb70", "pdb_mmcif", "uniref90", 
        "mgnify", "bfd", "uniclust30"
    ]
    
    for db_dir in db_dirs:
        os.makedirs(data_dir / db_dir, exist_ok=True)
        
        # Create minimal placeholder files
        placeholder_file = data_dir / db_dir / "placeholder.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"This is a placeholder for the {db_dir} database.\n")
            f.write("This setup is for testing only and does not contain actual database files.\n")
    
    logger.info("Minimal database setup complete")
    return data_dir

def validate_setup(alphafold_dir, data_dir):
    """Validate the AlphaFold setup."""
    issues = []
    
    # Check AlphaFold directory
    if not (alphafold_dir / "run_alphafold.py").exists():
        issues.append(f"AlphaFold run script not found at {alphafold_dir / 'run_alphafold.py'}")
    
    # Check data directories
    required_dirs = ["params", "pdb70", "pdb_mmcif", "uniref90", "mgnify", "bfd", "uniclust30"]
    for dir_name in required_dirs:
        if not (data_dir / dir_name).exists():
            issues.append(f"Required data directory {dir_name} not found at {data_dir / dir_name}")
    
    # Report issues
    if issues:
        logger.warning("AlphaFold setup validation found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("AlphaFold setup validation successful")
    return True

def main():
    parser = argparse.ArgumentParser(description="Set up AlphaFold for Protein Analysis Platform")
    parser.add_argument("--install-dir", type=str, default="/opt",
                        help="Directory to install AlphaFold (default: /opt)")
    parser.add_argument("--data-dir", type=str, default="/opt/alphafold_data",
                        help="Directory for AlphaFold data (default: /opt/alphafold_data)")
    parser.add_argument("--minimal", action="store_true",
                        help="Set up minimal placeholder databases for testing")
    parser.add_argument("--skip-repo", action="store_true",
                        help="Skip setting up AlphaFold repository")
    parser.add_argument("--skip-db", action="store_true",
                        help="Skip downloading databases")
    
    args = parser.parse_args()
    
    # Set up AlphaFold repository
    if not args.skip_repo:
        alphafold_dir = setup_alphafold_repo(args.install_dir)
    else:
        alphafold_dir = Path(args.install_dir) / "alphafold"
        logger.info(f"Skipping AlphaFold repository setup, using {alphafold_dir}")
    
    # Set up databases
    if not args.skip_db:
        if args.minimal:
            data_dir = setup_minimal_databases(args.data_dir)
        else:
            data_dir = download_databases(args.data_dir)
    else:
        data_dir = Path(args.data_dir)
        logger.info(f"Skipping database setup, using {data_dir}")
    
    # Validate setup
    validate_setup(alphafold_dir, data_dir)
    
    logger.info("AlphaFold setup complete")
    logger.info(f"AlphaFold repository: {alphafold_dir}")
    logger.info(f"AlphaFold data: {data_dir}")

if __name__ == "__main__":
    main()