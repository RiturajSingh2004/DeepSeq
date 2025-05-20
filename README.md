# DeepSeq - Advance Protein Analysis

<p align="center"><img src="logo.png" alt="DeepSeq Logo" width="350" /></p>

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.22.0+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: ESMFold](https://img.shields.io/badge/Model-ESMFold-green.svg)](https://esmatlas.com)
[![Visualization: Py3DMol](https://img.shields.io/badge/Visualization-Py3DMol-purple.svg)](https://3dmol.csb.pitt.edu)

AI-powered protein analysis platform leveraging state-of-the-art machine learning for structure prediction and functional insights.

[Installation](#-installation) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

---

</div>

## Overview

DeepSeq combines ESMFold's structure prediction capabilities with comprehensive protein analysis tools in an intuitive interface. Key features:

🧬 **Structure & Analysis**
- Structure prediction via ESMFold
- Sequence embeddings & property analysis
- Stability & function prediction

🎯 **Applications**
- Protein engineering & design
- Structure-function relationships
- Mutation impact analysis
- Drug discovery support

⚡ **Key Benefits**
- Real-time interactive analysis
- Multiple visualization modes
- Batch processing support
- Extensive API access

## Getting Started

```bash
# Clone repository
git clone https://github.com/riturajsingh/deepseq.git
cd deepseq

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

Visit `http://localhost:8501` in your browser to start analyzing proteins.

## Documentation

### Basic Usage

```python
from deepseq import analyze_protein, render_structure

# Analyze single protein
sequence = "MVKVGVNG..."
results = analyze_protein(sequence, mode='all')

# Visualize structure
render_structure(results['structure'], style='cartoon')

# Access properties
print(f"Stability Score: {results['stability']:.2f}")
print(f"Predicted Location: {results['predictions']['location']}")
```

### Advanced Features

#### MoleculeRenderer (`render_mol`)

Provides interactive 3D visualization of protein structures.

**Key Functions:**
- Customizable visualization styles (cartoon, stick, sphere)
- Multiple color schemes
- Interactive rotation and zoom
- Adjustable background color
- Optional spinning animation

**Benefits:**
- Interactive molecular visualization
- Multiple representation styles
- Easy-to-use interface
- High-quality rendering

### Property Analysis

#### ResidueAnalyzer

Analyzes per-residue properties of proteins.

**Key Functions:**
- `predict_residue_properties()`: Analyzes hydrophobicity, charge, and other properties
- `estimate_stability()`: Predicts protein stability under different conditions
- `predict_protein_properties()`: Assesses localization and functional properties
- `validate_sequence()`: Ensures valid amino acid sequences

**Benefits:**
- Comprehensive property analysis
- Stability prediction under various conditions
- Functional annotation
- Input validation

### ML Models

#### EmbeddingGenerator

Generates protein sequence embeddings for analysis.

**Key Functions:**
- `generate_embeddings()`: Creates sequence and residue-level embeddings
- `reduce_dimensions()`: Reduces embedding dimensionality for visualization
- Supports both PCA and t-SNE dimensionality reduction

**Benefits:**
- High-dimensional sequence representation
- Captures protein characteristics
- Enables similarity analysis
- Visualization-ready output

## 📈 How It Works

### Workflow

1. **Sequence Input**:
   - Users enter protein sequence in FASTA format
   - Sequence validation ensures valid amino acid codes
   - Option to use example sequence for testing

2. **Analysis Mode Selection**:
   - Overview
   - Structure Prediction
   - Sequence Embeddings
   - Residue Properties
   - Stability Analysis
   - Function Prediction

3. **Property Analysis**:
   - Calculates per-residue properties
   - Generates hydrophobicity profiles
   - Analyzes charge distribution
   - Predicts secondary structure propensity

4. **Structure Prediction**:
   - Submits sequence to ESMFold API
   - Receives predicted 3D structure
   - Calculates confidence scores
   - Enables structure visualization

5. **Visualization**:
   - Interactive 3D structure display
   - Property plots and charts
   - Embedding visualizations
   - Radar charts for stability

### Technical Details

The application uses several advanced technologies:

1. **ESMFold Integration**:
   - API calls to ESMFold service
   - PDB structure processing
   - Confidence score calculation
   - Structure quality assessment

2. **Visualization Pipeline**:
   - Py3Dmol for structure visualization
   - Matplotlib for property plots
   - Streamlit for interactive widgets
   - Custom color schemes

3. **Property Calculations**:
   - Amino acid property database
   - Physical property calculations
   - Statistical analysis
   - Machine learning predictions

## 📋 Advanced Usage Guide

### API Reference

#### Core Functions

```python
def analyze_protein(sequence: str, mode: str = 'all') -> dict:
    """
    Comprehensive protein analysis function.
    
    Args:
        sequence (str): Amino acid sequence in single-letter code
        mode (str): Analysis mode ('all', 'structure', 'properties', 'embeddings')
    
    Returns:
        dict: Analysis results including structure, properties, and predictions
    """
    results = {
        'sequence': sequence,
        'properties': {},
        'structure': None,
        'embeddings': None,
        'predictions': {}
    }
    # ... implementation details

def predict_residue_properties(sequence: str) -> dict:
    """
    Predict per-residue biochemical properties.
    
    Args:
        sequence (str): Amino acid sequence
    
    Returns:
        dict: Properties including hydrophobicity, charge, accessibility
    """
    properties = {
        'hydrophobicity': [],
        'charge': [],
        'helix_propensity': [],
        'sheet_propensity': [],
        'solvent_accessibility': []
    }
    # ... implementation details

def estimate_stability(
    sequence: str, 
    temperature: float = 25.0,
    ph: float = 7.0,
    salt_concentration: float = 0.15
) -> float:
    """
    Estimate protein stability under given conditions.
    
    Args:
        sequence (str): Amino acid sequence
        temperature (float): Temperature in Celsius
        ph (float): pH value
        salt_concentration (float): Salt concentration in M
    
    Returns:
        float: Stability score (0-10)
    """
    # ... implementation details

def render_mol(
    pdb: str,
    style: str = 'cartoon',
    color_scheme: str = 'spectrum',
    spin: bool = True,
    background: str = 'white'
) -> None:
    """
    Render protein structure using py3Dmol.
    
    Args:
        pdb (str): PDB structure string
        style (str): Visualization style
        color_scheme (str): Color scheme for visualization
        spin (bool): Enable structure rotation
        background (str): Background color
    """
    view = py3Dmol.view()
    view.addModel(pdb, 'pdb')
    # ... implementation details
```

### Advanced Examples

#### 1. Custom Property Analysis

```python
import numpy as np
from deepseq.analysis import predict_residue_properties
from deepseq.visualization import plot_property_profile

# Analyze multiple sequences
sequences = {
    "wild_type": "MVKVGVNG...",
    "mutant": "MVKVGVDG..."
}

# Compare properties
for name, seq in sequences.items():
    props = predict_residue_properties(seq)
    
    # Calculate property statistics
    stats = {
        'hydrophobicity_mean': np.mean(props['hydrophobicity']),
        'charge_sum': sum(props['charge']),
        'helix_content': np.mean(props['helix_propensity'])
    }
    
    print(f"{name} Analysis:")
    print(f"Mean Hydrophobicity: {stats['hydrophobicity_mean']:.2f}")
    print(f"Net Charge: {stats['charge_sum']}")
    print(f"Helix Content: {stats['helix_content']:.2%}")
```

#### 2. Structure Analysis Pipeline

```python
from deepseq.structure import fetch_structure, analyze_structure
from deepseq.visualization import render_structure, highlight_residues

# Fetch and analyze structure
def analyze_protein_structure(sequence: str, highlight_conserved: bool = True):
    # Get structure prediction
    structure = fetch_structure(sequence)
    
    # Basic structure analysis
    analysis = analyze_structure(structure)
    print(f"Structure Quality (plDDT): {analysis['plddt']:.2f}")
    
    # Identify conserved residues
    if highlight_conserved:
        conserved = find_conserved_residues(sequence)
        highlight_residues(structure, conserved, color='red')
    
    # Calculate accessible surface area
    asa = calculate_asa(structure)
    buried_residues = [i for i, a in enumerate(asa) if a < 20]
    print(f"Found {len(buried_residues)} buried residues")
    
    return structure, analysis

# Example usage
sequence = "MVKVGVNG..."
structure, analysis = analyze_protein_structure(sequence)
render_structure(structure, style='cartoon', colored_by='conservation')
```

#### 3. Batch Processing with Progress Tracking

```python
import pandas as pd
from tqdm import tqdm
from deepseq.batch import BatchProcessor
from deepseq.utils import validate_sequence

def batch_analyze_sequences(sequences: dict, modes: list = ['structure', 'properties']):
    """
    Batch analysis of multiple sequences with progress tracking.
    
    Args:
        sequences (dict): Dictionary of sequence names and sequences
        modes (list): Analysis modes to run
    
    Returns:
        pd.DataFrame: Results summary
    """
    results = []
    
    with tqdm(total=len(sequences)) as pbar:
        for name, seq in sequences.items():
            # Validate sequence
            is_valid, invalid_char = validate_sequence(seq)
            if not is_valid:
                print(f"Warning: Invalid sequence {name}, skipping...")
                continue
            
            # Run analysis
            try:
                analysis = analyze_protein(seq, modes=modes)
                results.append({
                    'name': name,
                    'length': len(seq),
                    'stability': analysis['stability'],
                    'location': analysis['predictions']['localization'],
                    'structure_confidence': analysis.get('plddt', None)
                })
            except Exception as e:
                print(f"Error analyzing {name}: {str(e)}")
            
            pbar.update(1)
    
    return pd.DataFrame(results)

# Example usage
sequences = {
    'protein1': 'MVKVGVNG...',
    'protein2': 'MAKLTFVG...',
    # ... more sequences
}

results_df = batch_analyze_sequences(sequences)
results_df.to_csv('analysis_results.csv')
```

### Command Line Interface

The package also provides a command-line interface for batch processing:

```bash
# Basic structure prediction
deepseq predict -i sequences.fasta -o predictions/

# Full analysis with all features
deepseq analyze -i sequences.fasta -o results/ --modes all

# Custom analysis with specific features
deepseq analyze -i sequences.fasta -o results/ `
    --modes structure,properties `
    --temp 37 `
    --ph 7.4 `
    --export-format json

# Batch processing with multiprocessing
deepseq batch -i sequences/ -o results/ `
    --workers 4 `
    --timeout 300 `
    --retry 3
```

### Configuration

DeepSeq can be configured using environment variables or a config file:

```yaml
# config.yaml
api:
  esmfold_endpoint: "https://api.esmatlas.com/foldSequence/v1/pdb/"
  timeout: 120
  retries: 3

analysis:
  default_modes:
    - structure
    - properties
    - embeddings
  cache_dir: "./cache"
  max_sequence_length: 1000

visualization:
  default_style: "cartoon"
  color_scheme: "spectrum"
  background: "white"
  spin: true

performance:
  batch_size: 32
  num_workers: 4
  use_gpu: true
```

### Web Interface Guide

The Streamlit interface provides an intuitive way to analyze proteins:

1. **Input Methods**:
   ```text
   a. Direct Sequence Input:
      - Paste FASTA sequence in sidebar
      - Support for multiple sequence formats
      - Automatic format detection
   
   b. File Upload:
      - Support for FASTA, PDB files
      - Batch processing capabilities
      - Progress tracking
   
   c. Example Sequences:
      - Pre-loaded examples for testing
      - Different protein classes
      - Validation cases
   ```

2. **Analysis Configuration**:
   ```text
   a. Mode Selection:
      - Overview: Quick summary
      - Structure: Detailed 3D analysis
      - Properties: Biochemical properties
      - Stability: Environmental conditions
      - Function: Predictions and annotations
   
   b. Visualization Options:
      - Style: cartoon, stick, sphere, surface
      - Color: spectrum, chain, residue type
      - Labels: residue numbers, atoms, chains
      - Effects: spinning, fog, shadows
   
   c. Analysis Parameters:
      - Temperature range: 0-100°C
      - pH range: 0-14
      - Salt concentration: 0-2M
      - Custom conditions
   ```

3. **Results and Export**:
   ```text
   a. Visualization:
      - Interactive 3D structure
      - Property plots
      - Radar charts
      - Heat maps
   
   b. Data Export:
      - PDB structures
      - Property CSV files
      - Analysis reports
      - Publication-ready figures
   
   c. Batch Results:
      - Summary tables
      - Comparative analysis
      - Statistical measures
      - Clustering results
   ```

## 🛣️ Roadmap

### Immediate Enhancements
- **Local Model Integration**: Add option to run ESM models locally
- **Batch Processing**: Enable analysis of multiple sequences
- **Enhanced Visualizations**: Add more plot types and interactivity
- **Property Predictions**: Implement more sophisticated prediction models

### Medium-term Goals
- **Database Integration**: Add protein database lookups
- **Alignment Features**: Implement sequence alignment tools
- **Export Options**: Add more export formats
- **API Development**: Create REST API for programmatic access

## 💻 Deployment

The application is designed for easy deployment:

- **Local Installation**: Run locally with minimal setup
- **Cloud Deployment**: Deploy on Streamlit Cloud
- **Docker Support**: Containerization for easy deployment
- **Resource Management**: Efficient memory usage

## ⚠️ Disclaimer

This tool is for research and educational purposes only. Predictions should be validated experimentally. Structure predictions are provided by the ESMFold service and subject to their terms of use.

## 📄 License

DeepSeq is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- **New Features**: Additional analysis tools
- **Visualization Enhancements**: New visualization options
- **Documentation**: Improved guides and examples
- **Performance**: Optimization and caching
- **Testing**: Unit tests and integration tests

## 👏 Acknowledgements

- [ESMFold](https://esmatlas.com) for structure prediction
- [Streamlit](https://streamlit.io/) for the web framework
- [Py3Dmol](https://3dmol.csb.pitt.edu) for molecular visualization
- [Matplotlib](https://matplotlib.org/) for plotting
- [scikit-learn](https://scikit-learn.org/) for data analysis
- [NumPy](https://numpy.org/) for numerical computations
- [Pandas](https://pandas.pydata.org/) for data manipulation

---

© 2025 Rituraj Singh | [GitHub](https://github.com/RiturajSingh2004) | [LinkedIn](https://linkedin.com/in/riturajsingh2004)
