# DeepSeq - Advanced Protein Analysis Tool

<p align="center"><img src="logo.png" alt="DeepSeq Logo" width="350" /></p>

<div align="center">
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.22.0+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: ESMFold](https://img.shields.io/badge/Model-ESMFold-green.svg)](https://esmatlas.com)
[![Visualization: Py3DMol](https://img.shields.io/badge/Visualization-Py3DMol-purple.svg)](https://3dmol.csb.pitt.edu)

AI-powered protein analysis tool that provides comprehensive structural and functional insights using state-of-the-art machine learning models.
</div>

## 📊 Features

- **3D Structure Prediction**: Accurate protein structure prediction using ESMFold
- **Sequence Embeddings**: High-dimensional vector representations of protein sequences
- **Property Analysis**: Detailed analysis of amino acid properties and protein characteristics
- **Interactive Visualizations**: 3D molecular visualization and property plots
- **Stability Assessment**: Predict protein stability under various conditions
- **Function Prediction**: Analyze subcellular localization, antimicrobial potential, and more
- **Real-time Analysis**: Instant feedback and interactive parameter adjustment
- **Custom Visualization Options**: Multiple viewing styles and color schemes

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip for package installation

### Installation

1. Clone the repository:
```bash
git clone https://github.com/riturajsingh/deepseq.git
cd deepseq
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

## 🔧 Project Structure

```
deepseq/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── logo.png               # Application logo
├── predicted.pdb          # Output structure file
└── README.md             # Documentation
```

## ⚙️ Core Components

### Visualization Module

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

## 📋 Usage Example

### Basic Analysis

```python
import streamlit as st
from stmol import showmol
import py3Dmol

# Initialize visualization
def render_mol(pdb, style='cartoon'):
    view = py3Dmol.view()
    view.addModel(pdb, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    showmol(view)

# Analyze sequence
sequence = "MVKVGVNG..."
results = analyze_protein(sequence)

# Display structure
if results['pdb_string']:
    render_mol(results['pdb_string'])

# Show properties
plot_residue_properties(results['properties'])
```

### Using the Streamlit Interface

1. Enter protein sequence:
   - Paste FASTA sequence in sidebar
   - Click "Validate Sequence" to check input
   - Use example sequence if needed

2. Select analysis mode:
   - Choose from available analysis types
   - Configure visualization options
   - Set stability parameters if needed

3. View results:
   - Explore 3D structure
   - Analyze property plots
   - Check stability predictions
   - Review function predictions

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
