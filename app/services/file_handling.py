import os
import logging
import tempfile
from typing import Dict, List, Optional, Union, BinaryIO, TextIO
from pathlib import Path
import shutil
import gzip
import zipfile
import uuid

from app.config import settings
from app.services.sequence_utils import extract_sequences_from_fasta, validate_sequence

# Configure logging
logger = logging.getLogger(__name__)

class FileHandler:
    """
    Handles file operations for protein sequence data.
    Processes various input formats including FASTA, raw text, and compressed files.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize file handler."""
        self.temp_dir = temp_dir or settings.TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_temp_directory(self) -> Path:
        """Create a temporary directory for file processing."""
        dir_uuid = uuid.uuid4().hex
        temp_path = self.temp_dir / dir_uuid
        temp_path.mkdir(parents=True, exist_ok=True)
        return temp_path
    
    def save_uploaded_file(self, file: BinaryIO, filename: str) -> Path:
        """
        Save an uploaded file to a temporary location.
        
        Args:
            file: File-like object
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        try:
            temp_dir = self._create_temp_directory()
            file_path = temp_dir / filename
            
            # Save the file
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file, f)
            
            logger.info(f"Saved uploaded file to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise
    
    def process_fasta_file(self, file_path: Path) -> Dict[str, str]:
        """
        Process a FASTA file and extract sequences.
        
        Args:
            file_path: Path to FASTA file
            
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        try:
            # Check if file is gzipped
            is_gzip = False
            if str(file_path).endswith('.gz'):
                is_gzip = True
            
            # Read file content
            if is_gzip:
                with gzip.open(file_path, 'rt') as f:
                    content = f.read()
            else:
                with open(file_path, 'r') as f:
                    content = f.read()
            
            # Extract sequences
            sequences = extract_sequences_from_fasta(content)
            
            if not sequences:
                logger.warning(f"No sequences found in FASTA file: {file_path}")
            else:
                logger.info(f"Extracted {len(sequences)} sequences from FASTA file")
            
            return sequences
        except Exception as e:
            logger.error(f"Error processing FASTA file: {e}")
            raise
    
    def process_text_file(self, file_path: Path) -> List[str]:
        """
        Process a text file containing sequences (one per line).
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of sequences
        """
        try:
            # Check if file is gzipped
            is_gzip = False
            if str(file_path).endswith('.gz'):
                is_gzip = True
            
            # Read file content
            if is_gzip:
                with gzip.open(file_path, 'rt') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
            
            # Process lines, skipping empty ones
            sequences = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip comments and empty lines
                    sequences.append(line)
            
            logger.info(f"Extracted {len(sequences)} sequences from text file")
            return sequences
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def process_zip_file(self, file_path: Path) -> Dict[str, Dict[str, str]]:
        """
        Process a ZIP file containing multiple sequence files.
        
        Args:
            file_path: Path to ZIP file
            
        Returns:
            Dictionary mapping filenames to sequence dictionaries
        """
        try:
            temp_dir = self._create_temp_directory()
            result = {}
            
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract to temporary directory
                zip_ref.extractall(temp_dir)
                
                # Process each file
                for extracted_file in temp_dir.iterdir():
                    if extracted_file.is_file():
                        file_extension = extracted_file.suffix.lower()
                        
                        if file_extension in ['.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn']:
                            # Process as FASTA
                            sequences = self.process_fasta_file(extracted_file)
                            result[extracted_file.name] = sequences
                        elif file_extension in ['.txt', '.seq']:
                            # Process as text
                            sequences = self.process_text_file(extracted_file)
                            result[extracted_file.name] = {f"seq_{i}": seq for i, seq in enumerate(sequences)}
            
            logger.info(f"Processed {len(result)} files from ZIP archive")
            return result
        except Exception as e:
            logger.error(f"Error processing ZIP file: {e}")
            raise
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def detect_file_type(self, file_path: Path) -> str:
        """
        Detect the type of a sequence file.
        
        Args:
            file_path: Path to file
            
        Returns:
            File type ('fasta', 'text', 'zip', 'unknown')
        """
        # Check by extension first
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn']:
            return 'fasta'
        elif file_extension == '.gz':
            # For gzipped files, check the original extension
            base_name = file_path.stem
            base_extension = Path(base_name).suffix.lower()
            if base_extension in ['.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn']:
                return 'fasta'
            elif base_extension in ['.txt', '.seq']:
                return 'text'
        elif file_extension in ['.txt', '.seq']:
            return 'text'
        elif file_extension == '.zip':
            return 'zip'
        
        # If extension is not conclusive, check content
        try:
            # Read first few lines
            if file_extension == '.gz':
                with gzip.open(file_path, 'rt') as f:
                    first_line = f.readline().strip()
            else:
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
            
            # Check if it looks like FASTA
            if first_line.startswith('>'):
                return 'fasta'
            
            # Check if it looks like a sequence
            if all(c.upper() in 'ACDEFGHIKLMNPQRSTVWY' for c in first_line):
                return 'text'
            
            # If we can't determine, default to text
            return 'unknown'
        except Exception:
            # If reading as text fails, it might be binary
            try:
                with open(file_path, 'rb') as f:
                    magic_bytes = f.read(4)
                
                # Check for ZIP signature: PK\x03\x04
                if magic_bytes.startswith(b'PK\x03\x04'):
                    return 'zip'
                
                # Check for gzip signature: \x1f\x8b
                if magic_bytes.startswith(b'\x1f\x8b'):
                    return 'gzip'
                
                return 'unknown'
            except Exception as e:
                logger.error(f"Error detecting file type: {e}")
                return 'unknown'
    
    def process_file(self, file_path: Path) -> Dict[str, str]:
        """
        Process a file with automatic type detection.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary mapping sequence IDs to sequences
        """
        file_type = self.detect_file_type(file_path)
        logger.info(f"Detected file type: {file_type} for {file_path}")
        
        if file_type == 'fasta':
            return self.process_fasta_file(file_path)
        elif file_type == 'text':
            sequences = self.process_text_file(file_path)
            return {f"seq_{i}": seq for i, seq in enumerate(sequences)}
        elif file_type == 'zip':
            # For ZIP, flatten the structure into a single dictionary
            zip_results = self.process_zip_file(file_path)
            flattened = {}
            for filename, sequences in zip_results.items():
                for seq_id, seq in sequences.items():
                    flattened[f"{filename}_{seq_id}"] = seq
            return flattened
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def validate_sequences(self, sequences: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate a dictionary of sequences.
        
        Args:
            sequences: Dictionary mapping sequence IDs to sequences
            
        Returns:
            Dictionary mapping sequence IDs to validation results
        """
        validation_results = {}
        for seq_id, sequence in sequences.items():
            validation_results[seq_id] = validate_sequence(sequence)
        
        return validation_results
    
    def cleanup(self, file_path: Optional[Path] = None):
        """
        Clean up temporary files.
        
        Args:
            file_path: Specific file to delete, or None to clean all temp files
        """
        try:
            if file_path and file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                logger.info(f"Cleaned up {file_path}")
            elif file_path is None:
                # Clean up all files older than 1 day
                import time
                current_time = time.time()
                for path in self.temp_dir.glob("**/*"):
                    if path.is_file() and (current_time - path.stat().st_mtime) > 86400:
                        path.unlink()
                    elif path.is_dir() and path != self.temp_dir and len(list(path.iterdir())) == 0:
                        path.rmdir()
                logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Singleton instance
_file_handler_instance = None

def get_file_handler() -> FileHandler:
    """Get or create the file handler singleton."""
    global _file_handler_instance
    if _file_handler_instance is None:
        _file_handler_instance = FileHandler()
    return _file_handler_instance