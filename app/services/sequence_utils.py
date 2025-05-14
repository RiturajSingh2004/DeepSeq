import re
import logging
from typing import List, Optional, Dict, Tuple, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

# Standard amino acid one-letter codes
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Including ambiguous amino acid codes
EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBZX*-")

def validate_sequence(sequence: str) -> bool:
    """
    Validate if a string is a valid protein sequence.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        True if valid, False otherwise
    """
    if not sequence:
        logger.warning("Empty sequence provided")
        return False
    
    # Remove whitespace
    sequence = sequence.strip().upper()
    
    # Check for valid characters
    if not all(aa in EXTENDED_AA for aa in sequence):
        invalid_chars = set(sequence) - EXTENDED_AA
        logger.warning(f"Invalid amino acid characters in sequence: {invalid_chars}")
        return False
    
    # Check sequence length
    if len(sequence) < 5:
        logger.warning(f"Sequence too short: {len(sequence)} amino acids")
        return False
    
    if len(sequence) > 10000:
        logger.warning(f"Sequence too long: {len(sequence)} amino acids")
        return False
    
    # Check for excessive unknown amino acids
    unknown_count = sum(1 for aa in sequence if aa in 'BZX*-')
    if unknown_count / len(sequence) > 0.1:
        logger.warning(f"Too many unknown amino acids: {unknown_count / len(sequence):.2%}")
        return False
    
    return True

def process_sequence(sequence: str) -> str:
    """
    Process and standardize a protein sequence.
    
    Args:
        sequence: Raw protein sequence string
        
    Returns:
        Processed sequence
    """
    if not sequence:
        return ""
    
    # Remove whitespace and convert to uppercase
    sequence = re.sub(r'\s+', '', sequence).upper()
    
    # Replace non-standard amino acids with X
    sequence = ''.join([aa if aa in STANDARD_AA else 'X' for aa in sequence])
    
    return sequence

def extract_sequences_from_fasta(fasta_content: str) -> Dict[str, str]:
    """
    Extract sequences from FASTA format string.
    
    Args:
        fasta_content: String containing FASTA formatted sequences
        
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    for line in fasta_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
            
            # Start new sequence
            current_id = line[1:].split()[0]  # Use first word as ID
            current_seq = []
        else:
            # Add line to current sequence
            current_seq.append(line)
    
    # Save last sequence
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    
    return sequences

def generate_fasta(sequences: Dict[str, str]) -> str:
    """
    Generate FASTA format from sequences.
    
    Args:
        sequences: Dictionary mapping sequence IDs to sequences
        
    Returns:
        FASTA formatted string
    """
    fasta_lines = []
    
    for seq_id, sequence in sequences.items():
        fasta_lines.append(f">{seq_id}")
        
        # Wrap sequence at 80 characters
        for i in range(0, len(sequence), 80):
            fasta_lines.append(sequence[i:i+80])
    
    return '\n'.join(fasta_lines)

def calculate_sequence_stats(sequence: str) -> Dict[str, Union[int, float]]:
    """
    Calculate basic sequence statistics.
    
    Args:
        sequence: Protein sequence string
        
    Returns:
        Dictionary of sequence statistics
    """
    # Process sequence
    sequence = process_sequence(sequence)
    
    if not sequence:
        return {
            'length': 0,
            'molecular_weight': 0.0,
            'isoelectric_point': 0.0,
            'amino_acid_composition': {},
            'charge_at_ph7': 0.0
        }
    
    # Calculate length
    length = len(sequence)
    
    # Count amino acids
    aa_counts = {}
    for aa in STANDARD_AA:
        aa_counts[aa] = sequence.count(aa)
    
    # Calculate amino acid composition (%)
    aa_composition = {aa: count / length * 100 for aa, count in aa_counts.items()}
    
    # Approximate molecular weight (Da)
    # Average amino acid weight ~110 Da
    mol_weight = sum(aa_counts.values()) * 110.0
    
    # Approximate isoelectric point using pKa of charged amino acids
    # This is a very simplified approximation
    acidic = aa_counts['D'] + aa_counts['E']
    basic = aa_counts['K'] + aa_counts['R'] + aa_counts['H']
    
    if acidic > basic:
        isoelectric_point = 5.5
    elif basic > acidic:
        isoelectric_point = 8.5
    else:
        isoelectric_point = 7.0
    
    # Approximate charge at pH 7
    charge_at_ph7 = basic - acidic
    
    return {
        'length': length,
        'molecular_weight': mol_weight,
        'isoelectric_point': isoelectric_point,
        'amino_acid_composition': aa_composition,
        'charge_at_ph7': charge_at_ph7
    }

def get_sequence_fragments(sequence: str, fragment_size: int = 10, overlap: int = 0) -> List[str]:
    """
    Split a sequence into overlapping fragments.
    
    Args:
        sequence: Protein sequence string
        fragment_size: Size of each fragment
        overlap: Number of overlapping residues between fragments
        
    Returns:
        List of sequence fragments
    """
    if fragment_size <= overlap:
        raise ValueError("Fragment size must be greater than overlap")
    
    sequence = process_sequence(sequence)
    fragments = []
    
    step = fragment_size - overlap
    for i in range(0, len(sequence) - fragment_size + 1, step):
        fragments.append(sequence[i:i+fragment_size])
    
    # Add the last fragment if it's not covered
    if len(sequence) % step != 0:
        last_start = (len(sequence) // step) * step
        if last_start + fragment_size <= len(sequence):
            fragments.append(sequence[last_start:last_start+fragment_size])
        else:
            fragments.append(sequence[last_start:])
    
    return fragments

def get_sequence_from_uniprot(uniprot_id: str) -> Optional[str]:
    """
    Retrieve a protein sequence from UniProt by ID.
    This is a simplified version that would need to be expanded with
    proper API calls in a real implementation.
    
    Args:
        uniprot_id: UniProt ID
        
    Returns:
        Protein sequence if found, None otherwise
    """
    try:
        import requests
        
        # Use UniProt API to retrieve the sequence
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url)
        
        if response.status_code == 200:
            fasta_content = response.text
            sequences = extract_sequences_from_fasta(fasta_content)
            
            if sequences:
                # Return the first sequence
                return list(sequences.values())[0]
        else:
            logger.warning(f"Failed to retrieve UniProt sequence: {response.status_code}")
        
        return None
    except Exception as e:
        logger.error(f"Error retrieving UniProt sequence: {e}")
        return None

def find_motifs(sequence: str, motif_patterns: Dict[str, str]) -> Dict[str, List[Tuple[int, str]]]:
    """
    Find sequence motifs in a protein sequence.
    
    Args:
        sequence: Protein sequence string
        motif_patterns: Dictionary mapping motif names to regex patterns
        
    Returns:
        Dictionary mapping motif names to lists of (position, matched_sequence) tuples
    """
    sequence = process_sequence(sequence)
    results = {}
    
    for motif_name, pattern in motif_patterns.items():
        matches = []
        for match in re.finditer(pattern, sequence):
            matches.append((match.start(), match.group()))
        
        results[motif_name] = matches
    
    return results

def calculate_hydropathy_profile(sequence: str, window_size: int = 7) -> List[float]:
    """
    Calculate Kyte-Doolittle hydropathy profile.
    
    Args:
        sequence: Protein sequence string
        window_size: Window size for smoothing
        
    Returns:
        List of hydropathy values
    """
    sequence = process_sequence(sequence)
    
    # Kyte-Doolittle hydrophobicity scale
    hydrophobicity = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
        'X': 0.0  # Unknown amino acid
    }
    
    # Calculate hydropathy for each residue
    values = [hydrophobicity.get(aa, 0.0) for aa in sequence]
    
    # Apply window averaging
    profile = []
    half_window = window_size // 2
    
    for i in range(len(sequence)):
        start = max(0, i - half_window)
        end = min(len(sequence), i + half_window + 1)
        profile.append(sum(values[start:end]) / (end - start))
    
    return profile