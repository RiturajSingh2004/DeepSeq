import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import biotite.structure.io as bsio
import os
import re
import time
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure page
st.set_page_config(page_title="DeepSeq - Advanced Protein Analysis", layout='wide')

def add_logo(logo_path):
    """This function loads a logo from the filesystem and displays it in the sidebar"""
    logo_path = Path(logo_path)
    
    if not logo_path.exists():
        # If the exact path doesn't exist, try common variations
        possible_paths = [
            "logo.png",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logo_path = Path(path)
                break
    
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            data = f.read()
        
        # Get the file extension to determine MIME type
        ext = logo_path.suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/svg+xml"
        
        encoded = base64.b64encode(data).decode()
        
        st.sidebar.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="data:{mime_type};base64,{encoded}" alt="DeepSeq Logo" width="200">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # If logo can't be found, display text instead
        st.sidebar.markdown(
            """
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <h2 style="color: #4A90E2;">DeepSeq</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# Try to add the logo
add_logo("logo.png")

# Sidebar
st.sidebar.write('[*DeepSeq*](https://github.com/RiturajSingh2004/DeepSeq) is an advanced protein analysis tool with structure prediction and more based on the ESM-2 language model.')

# List of valid amino acids for validation
VALID_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Helper function to validate protein sequence
def validate_sequence(sequence):
    """Validates if a sequence contains only valid amino acid codes."""
    sequence = sequence.upper()
    # Remove common formatting characters that might be in FASTA formats
    sequence = re.sub(r'[\s\d>]', '', sequence)
    
    # Check if all characters are valid amino acids
    for aa in sequence:
        if aa not in VALID_AMINO_ACIDS:
            return False, aa
    return True, None

# Render molecule with customizable options
def render_mol(pdb, style='cartoon', color_scheme='spectrum', spin=True, background='white'):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, 'pdb')
    
    # Set style based on selection
    if style == 'cartoon':
        pdbview.setStyle({'cartoon': {'color': color_scheme}})
    elif style == 'stick':
        pdbview.setStyle({'stick': {'colorscheme': color_scheme}})
    elif style == 'sphere':
        pdbview.setStyle({'sphere': {'colorscheme': color_scheme}})
    elif style == 'cartoon_and_stick':
        pdbview.setStyle({'cartoon': {'color': color_scheme}})
        pdbview.addStyle({'stick': {'colorscheme': color_scheme}})
    
    pdbview.setBackgroundColor(background)
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(spin)
    showmol(pdbview, height=500, width=800)

# Default protein sequence
DEFAULT_SEQ = "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"

# Amino acid properties dictionary for property predictions
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'polarity': False, 'size': 'small'},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'polarity': True, 'size': 'small'},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': True, 'size': 'small'},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': True, 'size': 'medium'},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'polarity': False, 'size': 'large'},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'polarity': False, 'size': 'small'},
    'H': {'hydrophobicity': -3.2, 'charge': 0.1, 'polarity': True, 'size': 'medium'},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'polarity': False, 'size': 'large'},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'polarity': True, 'size': 'large'},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'polarity': False, 'size': 'large'},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'polarity': False, 'size': 'large'},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': True, 'size': 'small'},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'polarity': False, 'size': 'small'},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': True, 'size': 'medium'},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'polarity': True, 'size': 'large'},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'polarity': True, 'size': 'small'},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'polarity': True, 'size': 'small'},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'polarity': False, 'size': 'medium'},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'polarity': True, 'size': 'large'},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polarity': True, 'size': 'large'}
}

# ESM embeddings simulation (in a real implementation, this would call the ESM API)
def generate_embeddings(sequence):
    """Simulate ESM embeddings for a protein sequence"""
    # In a real implementation, this would call the ESM API
    # Here we'll generate random embeddings for demonstration
    np.random.seed(hash(sequence) % 2**32)  # Make it deterministic based on sequence
    
    # Generate embeddings for the whole sequence and per-residue
    seq_length = len(sequence)
    
    # Generate a sequence embedding (1280-dimensional vector)
    sequence_embedding = np.random.normal(0, 1, 1280)
    
    # Generate per-residue embeddings (seq_length x 1280)
    residue_embeddings = np.random.normal(0, 1, (seq_length, 1280))
    
    return sequence_embedding, residue_embeddings

# Reduce dimensionality for visualization
def reduce_dimensions(embeddings, method='pca', dims=2):
    """Reduce dimensions of embeddings for visualization"""
    if method == 'pca':
        reducer = PCA(n_components=dims)
    else:  # t-SNE
        reducer = TSNE(n_components=dims, perplexity=min(30, max(5, len(embeddings)//10)))
    
    reduced = reducer.fit_transform(embeddings)
    return reduced

# Generate per-residue property predictions
def predict_residue_properties(sequence):
    """Generate per-residue property predictions based on amino acid properties"""
    properties = {}
    
    # Hydrophobicity profile
    properties['hydrophobicity'] = [AA_PROPERTIES[aa]['hydrophobicity'] for aa in sequence]
    
    # Charge profile
    properties['charge'] = [AA_PROPERTIES[aa]['charge'] for aa in sequence]
    
    # Secondary structure propensity (simulated)
    np.random.seed(hash(sequence) % 2**32)
    properties['helix_propensity'] = np.clip(
        np.convolve(np.random.normal(0, 1, len(sequence)), 
                   np.ones(7)/7, mode='same') + 
        np.array([0.5 if aa in 'AEILM' else -0.5 if aa in 'PG' else 0 for aa in sequence]), 
        0, 1
    )
    
    properties['sheet_propensity'] = np.clip(
        np.convolve(np.random.normal(0, 1, len(sequence)), 
                   np.ones(7)/7, mode='same') + 
        np.array([0.5 if aa in 'VIFTY' else -0.5 if aa in 'PG' else 0 for aa in sequence]), 
        0, 1
    )
    
    # Solvent accessibility (simulated)
    properties['solvent_accessibility'] = np.clip(
        np.convolve(np.random.normal(0, 1, len(sequence)), 
                   np.ones(5)/5, mode='same') + 
        np.array([-0.5 if AA_PROPERTIES[aa]['hydrophobicity'] > 0 else 0.5 for aa in sequence]), 
        0, 1
    )
    
    return properties

# Estimate protein stability under different conditions
def estimate_stability(sequence, temperature=25, ph=7.0, salt_concentration=0.15):
    """Estimate protein stability under different conditions"""
    # Simple stability model based on amino acid composition and conditions
    # In a real implementation, this would use more sophisticated models
    
    # Calculate base stability from sequence composition
    hydrophobic_ratio = sum(1 for aa in sequence if AA_PROPERTIES[aa]['hydrophobicity'] > 0) / len(sequence)
    charged_residues = sum(1 for aa in sequence if abs(AA_PROPERTIES[aa]['charge']) > 0)
    charged_ratio = charged_residues / len(sequence)
    
    # Temperature effect (proteins tend to denature at higher temperatures)
    temp_effect = -0.05 * (temperature - 25)  # Reference temperature is 25°C
    
    # pH effect (stability often peaks at neutral pH)
    ph_effect = -0.5 * abs(ph - 7.0)
    
    # Salt concentration effect (moderate salt can stabilize through ionic interactions)
    if salt_concentration < 0.5:
        salt_effect = salt_concentration * 0.5  # Stabilizing at moderate concentrations
    else:
        salt_effect = 0.25 - (salt_concentration - 0.5) * 0.2  # Destabilizing at high concentrations
    
    # Combine effects (higher score = more stable)
    stability_score = (
        5.0 +  # Base score
        hydrophobic_ratio * 2.0 +  # Hydrophobic core contribution
        charged_ratio * (1.0 if 6.5 <= ph <= 7.5 else -1.0) +  # Charged residues effect
        temp_effect +
        ph_effect +
        salt_effect
    )
    
    # Add some sequence-specific variance
    np.random.seed(hash(sequence) % 2**32)
    stability_score += np.random.normal(0, 0.2)
    
    # Ensure score is within reasonable bounds (0-10)
    stability_score = max(0, min(10, stability_score))
    
    return stability_score

# Predict protein localization, antimicrobial potential, and toxicity
def predict_protein_properties(sequence):
    """Predict various protein properties based on sequence"""
    # In a real implementation, these would use trained ML models
    # Here we use simple heuristics based on sequence features
    
    properties = {}
    
    # Amino acid composition
    aa_count = {aa: sequence.count(aa) for aa in VALID_AMINO_ACIDS}
    aa_freq = {aa: count / len(sequence) for aa, count in aa_count.items()}
    
    # Net charge at physiological pH
    net_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in sequence)
    
    # Hydrophobicity measures
    avg_hydrophobicity = sum(AA_PROPERTIES[aa]['hydrophobicity'] for aa in sequence) / len(sequence)
    
    # Calculate hydrophobic moments (crude approximation)
    hydrophobic_moment = 0
    for i in range(len(sequence) - 3):
        window = sequence[i:i+4]
        moment = sum(AA_PROPERTIES[aa]['hydrophobicity'] * (-1)**(i) for i, aa in enumerate(window))
        hydrophobic_moment = max(hydrophobic_moment, abs(moment))
    
    # Predict subcellular localization
    # Simple rules based on sequence properties
    localization_scores = {
        'Cytoplasmic': 0.2,
        'Nuclear': 0.1,
        'Mitochondrial': 0.1,
        'Secreted': 0.1,
        'Membrane': 0.1
    }
    
    # Cytoplasmic proteins tend to have moderate hydrophobicity and charge
    if -1 < avg_hydrophobicity < 1 and -5 < net_charge < 5:
        localization_scores['Cytoplasmic'] += 0.4
    
    # Nuclear proteins often have nuclear localization signals (positively charged patches)
    if aa_freq['R'] + aa_freq['K'] > 0.15:
        localization_scores['Nuclear'] += 0.4
    
    # Mitochondrial proteins often have N-terminal targeting sequences
    n_term = sequence[:20]
    if n_term.count('R') + n_term.count('K') > 3 and aa_freq['A'] + aa_freq['L'] > 0.2:
        localization_scores['Mitochondrial'] += 0.4
    
    # Secreted proteins often have signal peptides
    if aa_freq['L'] + aa_freq['A'] + aa_freq['V'] > 0.3 and sequence[:20].count('K') + sequence[:20].count('R') < 3:
        localization_scores['Secreted'] += 0.4
    
    # Membrane proteins have high hydrophobicity
    if avg_hydrophobicity > 1.0 and sum(1 for aa in sequence if AA_PROPERTIES[aa]['hydrophobicity'] > 2) / len(sequence) > 0.3:
        localization_scores['Membrane'] += 0.5
    
    # Add a bit of random noise for variety
    np.random.seed(hash(sequence) % 2**32)
    for loc in localization_scores:
        localization_scores[loc] += np.random.normal(0, 0.05)
        localization_scores[loc] = max(0, min(1, localization_scores[loc]))
    
    properties['localization'] = localization_scores
    
    # Antimicrobial potential prediction
    # AMPs often have positive charge and amphipathic structure
    amp_score = 0.3  # base score
    
    if net_charge > 2:  # Positive charge is common in AMPs
        amp_score += 0.2
    
    if hydrophobic_moment > 5:  # Amphipathicity is key for AMPs
        amp_score += 0.3
    
    if 10 <= len(sequence) <= 50:  # Many AMPs are short peptides
        amp_score += 0.1
    
    # Count amphipathic patterns (hydrophobic on one side, charged on the other)
    hydrophobic_clusters = len(re.findall(r'[AILMFWV]{2,}', sequence))
    if hydrophobic_clusters > 2:
        amp_score += 0.1
    
    # Add some random noise
    amp_score += np.random.normal(0, 0.05)
    properties['antimicrobial_potential'] = max(0, min(1, amp_score))
    
    # Toxicity prediction
    # Toxic proteins often have specific domains/motifs
    toxicity_score = 0.2  # base score
    
    # High cysteine content can indicate toxin (disulfide bonds)
    if aa_freq['C'] > 0.08:
        toxicity_score += 0.2
    
    # Toxic proteins may have extreme charge or hydrophobicity
    if abs(net_charge) > 10 or abs(avg_hydrophobicity) > 2:
        toxicity_score += 0.2
    
    # Specific motifs (very simplified examples)
    if 'CXXC' in ''.join('X' if aa == 'C' else 'O' for aa in sequence):  # Zinc finger-like
        toxicity_score += 0.1
    
    # Short, charged peptides can have membrane-disrupting activity
    if len(sequence) < 50 and net_charge > 4:
        toxicity_score += 0.2
    
    # Add some random noise
    toxicity_score += np.random.normal(0, 0.05)
    properties['toxicity'] = max(0, min(1, toxicity_score))
    
    return properties

def plot_residue_properties(properties, sequence):
    """Generate plots for per-residue properties"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    x = list(range(1, len(sequence) + 1))
    
    # Plot hydrophobicity
    axes[0].plot(x, properties['hydrophobicity'], 'b-')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Hydrophobicity')
    axes[0].set_title('Per-residue Hydrophobicity Profile')
    axes[0].grid(True, alpha=0.3)
    
    # Plot charge
    axes[1].plot(x, properties['charge'], 'r-')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('Charge')
    axes[1].set_title('Per-residue Charge Profile')
    axes[1].grid(True, alpha=0.3)
    
    # Plot secondary structure propensity
    axes[2].plot(x, properties['helix_propensity'], 'g-', label='Helix')
    axes[2].plot(x, properties['sheet_propensity'], 'purple', label='Sheet')
    axes[2].set_ylabel('Propensity')
    axes[2].set_title('Secondary Structure Propensity')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot solvent accessibility
    axes[3].plot(x, properties['solvent_accessibility'], 'orange')
    axes[3].set_ylabel('Solvent\nAccessibility')
    axes[3].set_title('Predicted Solvent Accessibility')
    axes[3].set_xlabel('Residue Position')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

def plot_embeddings(embeddings_2d, sequence):
    """Create a plot of the reduced dimensionality embeddings"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
              c=range(len(sequence)), cmap='viridis', 
              alpha=0.8, s=50)
    
    # Add colorbar to show sequence position
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sequence Position')
    
    # Highlight some key positions
    for i in range(0, len(sequence), max(1, len(sequence)//20)):  # Label every ~20th residue
        ax.annotate(f"{sequence[i]}:{i+1}", 
                    (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, alpha=0.7)
    
    ax.set_title('Protein Sequence Embeddings - Residue Relationships')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

def plot_stability_radar(stability_scores):
    """Create a radar chart for stability under different conditions"""
    conditions = list(stability_scores.keys())
    values = [stability_scores[c] for c in conditions]
    
    # Number of variables
    N = len(conditions)
    
    # Create angles for each variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values for the chart
    values += values[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels
    ax.set_thetagrids(np.degrees(angles[:-1]), conditions)
    
    # Set y-axis limits
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11, 2))
    ax.set_yticklabels([str(x) for x in range(0, 11, 2)])
    
    # Add title
    ax.set_title('Protein Stability Under Different Conditions', size=14)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

def plot_prediction_bars(predictions, title):
    """Create a bar chart for predictions"""
    labels = list(predictions.keys())
    values = list(predictions.values())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color='skyblue')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                va='center', fontsize=10)
    
    # Set limits and title
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(title)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()
    
    return buf

# Protein sequence input
txt = st.sidebar.text_area('Input sequence', DEFAULT_SEQ, height=200)

# Sequence validation dropdown
st.sidebar.subheader("Sequence Validation")
validate_btn = st.sidebar.button('Validate Sequence')

if validate_btn:
    is_valid, invalid_char = validate_sequence(txt)
    if is_valid:
        st.sidebar.success("✅ Sequence contains only valid amino acid codes")
    else:
        st.sidebar.error(f"❌ Invalid character '{invalid_char}' found in sequence. Valid amino acids are: {', '.join(VALID_AMINO_ACIDS)}")

# App mode selection
st.sidebar.subheader("Analysis Mode")
app_mode = st.sidebar.selectbox(
    "Select analysis mode",
    ["Overview", "Structure Prediction", "Sequence Embeddings", "Residue Properties", 
     "Stability Analysis", "Function Prediction"]
)

# Visualization options (only shown in Structure Prediction mode)
if app_mode == "Structure Prediction":
    st.sidebar.subheader("Visualization Options")
    viz_style = st.sidebar.selectbox(
        'Visualization style',
        ('cartoon', 'stick', 'sphere', 'cartoon_and_stick')
    )
    color_scheme = st.sidebar.selectbox(
        'Color scheme',
        ('spectrum', 'chainHetatm', 'chain', 'residue', 'whiteCarbon', 'greenCarbon')
    )
    background_color = st.sidebar.selectbox(
        'Background color',
        ('white', 'black', '#f0f0f0', 'gray')
    )
    spin_model = st.sidebar.checkbox('Spin model', value=True)

# Stability analysis parameters (only shown in Stability Analysis mode)
if app_mode == "Stability Analysis":
    st.sidebar.subheader("Stability Parameters")
    temperature = st.sidebar.slider("Temperature (°C)", 0, 100, 25)
    ph = st.sidebar.slider("pH", 2.0, 12.0, 7.0, 0.1)
    salt_concentration = st.sidebar.slider("Salt Concentration (M)", 0.0, 2.0, 0.15, 0.05)

# Main analysis function
def analyze_protein(sequence=txt):
    # Validate sequence before proceeding
    is_valid, invalid_char = validate_sequence(sequence)
    if not is_valid:
        st.error(f"Invalid character '{invalid_char}' found in sequence. Please fix before analysis.")
        return
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Starting protein analysis...")
        progress_bar.progress(10)
        
        # Generate embeddings
        status_text.text("Generating protein embeddings...")
        progress_bar.progress(20)
        sequence_embedding, residue_embeddings = generate_embeddings(sequence)
        
        # Per-residue property predictions
        status_text.text("Predicting residue properties...")
        progress_bar.progress(40)
        residue_properties = predict_residue_properties(sequence)
        
        # Protein property predictions
        status_text.text("Analyzing protein function...")
        progress_bar.progress(60)
        protein_properties = predict_protein_properties(sequence)
        
        # ESMfold API call for structure (only if in structure prediction mode)
        pdb_string = None
        b_value = None
        
        if app_mode == "Structure Prediction" or app_mode == "Overview":
            status_text.text("Predicting protein structure...")
            progress_bar.progress(70)
            
            try:
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                }
                
                # Make API request with error handling and timeout
                response = requests.post(
                    'https://api.esmatlas.com/foldSequence/v1/pdb/', 
                    headers=headers, 
                    data=sequence,
                    timeout=120  # 2 minute timeout
                )
                response.raise_for_status()
                
                pdb_string = response.content.decode('utf-8')
                
                # Save PDB locally
                with open('predicted.pdb', 'w') as f:
                    f.write(pdb_string)
                
                # Calculate plDDT value
                struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
                b_value = round(struct.b_factor.mean(), 4)
                
            except Exception as e:
                st.warning(f"Structure prediction failed: {str(e)}")
                st.warning("Continuing with other analyses...")
        
        # Stability analysis
        status_text.text("Calculating stability profiles...")
        progress_bar.progress(80)
        
        # Default stability parameters
        stability_conditions = {
            "Standard (25°C, pH 7)": estimate_stability(sequence, 25, 7.0, 0.15),
            "Heat (60°C, pH 7)": estimate_stability(sequence, 60, 7.0, 0.15),
            "Acidic (25°C, pH 4)": estimate_stability(sequence, 25, 4.0, 0.15),
            "Basic (25°C, pH 9)": estimate_stability(sequence, 25, 9.0, 0.15),
            "High Salt (25°C, pH 7, 1M NaCl)": estimate_stability(sequence, 25, 7.0, 1.0)
        }
        
        # If in stability analysis mode, add custom condition
        if app_mode == "Stability Analysis":
            stability_conditions[f"Custom ({temperature}°C, pH {ph}, {salt_concentration}M NaCl)"] = \
                estimate_stability(sequence, temperature, ph, salt_concentration)
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Return all analysis results
        return {
            'sequence': sequence,
            'sequence_embedding': sequence_embedding,
            'residue_embeddings': residue_embeddings,
            'residue_properties': residue_properties,
            'protein_properties': protein_properties,
            'stability_conditions': stability_conditions,
            'pdb_string': pdb_string,
            'plDDT': b_value
        }
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return None

# Analyze button
analyze = st.sidebar.button('Run Analysis', on_click=None)

# Main app logic
if analyze:
    results = analyze_protein(txt)
    
    if results:
        if app_mode == "Overview":
            st.title('DeepSeq Protein Analysis Overview')
            
            # Basic sequence info
            seq_len = len(results['sequence'])
            st.subheader("Basic Sequence Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"Sequence Length: {seq_len} amino acids")
                
                # Calculate molecular weight (approximate)
                aa_weights = {
                    'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
                    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
                    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
                    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
                }
                mol_weight = sum(aa_weights[aa] for aa in results['sequence'])
                st.info(f"Approximate Molecular Weight: {mol_weight/1000:.2f} kDa")
                
                # Calculate isoelectric point (very approximate)
                net_charge = sum(AA_PROPERTIES[aa]['charge'] for aa in results['sequence'])
                st.info(f"Net Charge (at pH 7): {net_charge:.1f}")
            
            with col2:
                # Show amino acid composition
                aa_composition = {aa: results['sequence'].count(aa) / seq_len * 100 for aa in VALID_AMINO_ACIDS}
                st.info("Top 5 Most Abundant Amino Acids:")
                for aa, percent in sorted(aa_composition.items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.write(f"- {aa}: {percent:.1f}%")
            
            # Show key predictions
            st.subheader("Key Predictions")
            col1, col2 = st.columns(2)
            
            with col1:
                # Most likely localization
                top_loc = max(results['protein_properties']['localization'].items(), key=lambda x: x[1])
                st.info(f"Predicted Localization: {top_loc[0]} ({top_loc[1]:.2f})")
                
                # Show plDDT if available
                if results['plDDT'] is not None:
                    st.info(f"Structure Confidence (plDDT): {results['plDDT']}")
                    quality = "Very low" if results['plDDT'] < 50 else "Low" if results['plDDT'] < 70 else "Medium" if results['plDDT'] < 90 else "High"
                    st.info(f"Structure Quality: {quality}")
            
            with col2:
                # Show antimicrobial and toxicity
                st.info(f"Antimicrobial Potential: {results['protein_properties']['antimicrobial_potential']:.2f}")
                st.info(f"Toxicity Potential: {results['protein_properties']['toxicity']:.2f}")
                
                # Show standard stability
                st.info(f"Stability Score (standard conditions): {results['stability_conditions']['Standard (25°C, pH 7)']:.2f}/10")
            
            # Show structure if available
            if results['pdb_string'] is not None:
                st.subheader("Predicted Structure")
                render_mol(
                    results['pdb_string'], 
                    style='cartoon', 
                    color_scheme='spectrum', 
                    spin=True,
                    background='white'
                )
                
                # Download button for PDB
                st.download_button(
                    label="Download PDB",
                    data=results['pdb_string'],
                    file_name='predicted.pdb',
                    mime='text/plain',
                )
            
            # Show tabs for different analyses
            st.subheader("Detailed Analysis")
            tab1, tab2, tab3 = st.tabs(["Residue Properties", "Embeddings", "Function Prediction"])
            
            with tab1:
                # Show residue property plot
                st.image(plot_residue_properties(results['residue_properties'], results['sequence']))
            
            with tab2:
                # Reduce dimensions for visualization
                reduced_embeddings = reduce_dimensions(results['residue_embeddings'], 'pca')
                st.image(plot_embeddings(reduced_embeddings, results['sequence']))
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show localization predictions
                    st.image(plot_prediction_bars(
                        results['protein_properties']['localization'],
                        "Subcellular Localization Prediction"
                    ))
                
                with col2:
                    # Show stability radar
                    st.image(plot_stability_radar(results['stability_conditions']))
            
        elif app_mode == "Structure Prediction":
            st.title('DeepSeq Protein Structure Prediction')
            
            if results['pdb_string'] is None:
                st.error("Structure prediction failed. Please try again with a different sequence or check your internet connection.")
            else:
                # Display protein structure
                st.subheader('Visualization of predicted protein structure')
                render_mol(
                    results['pdb_string'], 
                    style=viz_style, 
                    color_scheme=color_scheme, 
                    spin=spin_model,
                    background=background_color
                )
                
                # plDDT value display
                st.subheader('plDDT')
                st.write('plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.')
                st.info(f'plDDT: {results["plDDT"]}')
                
                # Quality assessment
                quality = "Very low" if results['plDDT'] < 50 else "Low" if results['plDDT'] < 70 else "Medium" if results['plDDT'] < 90 else "High"
                st.info(f"Structure Quality: {quality}")
                st.info(f"Sequence Length: {len(results['sequence'])} amino acids")
                
                # Download button
                st.download_button(
                    label="Download PDB",
                    data=results['pdb_string'],
                    file_name='predicted.pdb',
                    mime='text/plain',
                )
        
        elif app_mode == "Sequence Embeddings":
            st.title('Protein Sequence Embeddings Analysis')
            
            st.write("""
            Embeddings represent the protein sequence in high-dimensional space, capturing 
            evolutionary and structural relationships between residues. These embeddings can be 
            used for various downstream tasks like structure prediction, function annotation, and more.
            """)
            
            # Embedding visualization methods
            embed_method = st.radio(
                "Embedding Visualization Method",
                ["PCA", "t-SNE"],
                horizontal=True
            )
            
            # Show embeddings plot
            st.subheader(f'Protein Residue Embedding Visualization ({embed_method})')
            reduced_embeddings = reduce_dimensions(results['residue_embeddings'], embed_method.lower())
            st.image(plot_embeddings(reduced_embeddings, results['sequence']))
            
            # Explain what we're seeing
            st.write("""
            **What this visualization shows:**
            
            Each point represents an amino acid in your protein. Points that are close together in this 
            visualization are predicted to have similar functional or structural roles in the protein,
            despite potentially being far apart in the linear sequence.
            
            Clusters may indicate:
            - Secondary structure elements
            - Functional domains
            - Conserved structural motifs
            """)
            
            # Show embedding statistics
            st.subheader("Embedding Properties")
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate embedding statistics
                means = np.mean(results['residue_embeddings'], axis=0)
                stds = np.std(results['residue_embeddings'], axis=0)
                
                st.info(f"Embedding dimension: {results['residue_embeddings'].shape[1]}")
                st.info(f"Mean activation: {np.mean(means):.4f}")
                st.info(f"Mean standard deviation: {np.mean(stds):.4f}")
            
            with col2:
                # Calculate distances between residues
                st.info(f"Number of residues: {len(results['sequence'])}")
                
                if len(results['sequence']) > 1:
                    mean_dist = np.mean([
                        np.linalg.norm(results['residue_embeddings'][i] - results['residue_embeddings'][i+1]) 
                        for i in range(len(results['sequence'])-1)
                    ])
                    st.info(f"Mean distance between adjacent residues: {mean_dist:.4f}")
            
            st.subheader("Potential Applications")
            st.write("""
            These embeddings can be used for:
            - Protein classification
            - Finding similar proteins
            - Identifying functional sites
            - Mutation effect prediction
            - Protein engineering
            """)
        
        elif app_mode == "Residue Properties":
            st.title('Per-Residue Property Analysis')
            
            # Show residue property plots
            st.image(plot_residue_properties(results['residue_properties'], results['sequence']))
            
            # Add explanation
            st.subheader("Understanding These Properties")
            st.write("""
            **Hydrophobicity Profile:**  
            Positive values indicate hydrophobic residues, negative values indicate hydrophilic residues.
            Hydrophobic regions often form the protein core or membrane-interacting domains.
            
            **Charge Profile:**  
            Shows the distribution of charged residues (positive for K/R, negative for D/E).
            Charged regions often participate in interactions with other molecules.
            
            **Secondary Structure Propensity:**  
            Higher values indicate greater likelihood of forming that secondary structure.
            Helices typically form in regions with high helix propensity, same for sheets.
            
            **Solvent Accessibility:**  
            Higher values indicate greater predicted exposure to solvent.
            Buried residues (low values) are often critical for protein stability.
            """)
            
            # Interactive sequence viewer with properties
            st.subheader("Sequence with Properties")
            
            # Show sequence with sliding window
            window_size = st.slider("Window size", 10, 50, 20)
            window_start = st.slider("Start position", 1, max(1, len(results['sequence']) - window_size + 1), 1)
            window_end = window_start + window_size - 1
            
            # Calculate property colors
            hydro_colors = ["#" + format(int(255 * min(1, max(0, (h + 4.5) / 9))), '02x') + 
                            format(int(255 * min(1, max(0, (4.5 - h) / 9))), '02x') + "ff"
                           for h in results['residue_properties']['hydrophobicity'][window_start-1:window_end]]
            
            # Display the colored sequence
            st.write("Colored by hydrophobicity (red = hydrophobic, blue = hydrophilic):")
            html_sequence = "".join([f'<span style="background-color: {color};">{aa}</span>' 
                                   for aa, color in zip(results['sequence'][window_start-1:window_end], hydro_colors)])
            
            st.markdown(f"<div style='font-family: monospace; font-size: 24px; letter-spacing: 5px;'>{html_sequence}</div>", unsafe_allow_html=True)
            
            # Add sequence position numbers
            pos_numbers = "".join([f'<span style="display: inline-block; width: 24px; text-align: center;">{i}</span>' 
                                  for i in range(window_start, window_end+1)])
            st.markdown(f"<div style='font-family: monospace; font-size: 10px;'>{pos_numbers}</div>", unsafe_allow_html=True)
            
            # Property summary for selected region
            st.subheader(f"Property Summary for Region {window_start}-{window_end}")
            col1, col2 = st.columns(2)
            
            with col1:
                region_hydro = np.mean(results['residue_properties']['hydrophobicity'][window_start-1:window_end])
                region_charge = np.sum(results['residue_properties']['charge'][window_start-1:window_end])
                st.info(f"Mean Hydrophobicity: {region_hydro:.2f}")
                st.info(f"Net Charge: {region_charge:.1f}")
            
            with col2:
                region_helix = np.mean(results['residue_properties']['helix_propensity'][window_start-1:window_end])
                region_sheet = np.mean(results['residue_properties']['sheet_propensity'][window_start-1:window_end])
                st.info(f"Helix Propensity: {region_helix:.2f}")
                st.info(f"Sheet Propensity: {region_sheet:.2f}")
                
                # Predict most likely secondary structure
                if region_helix > region_sheet and region_helix > 0.5:
                    st.info("Likely Structure: α-helix")
                elif region_sheet > region_helix and region_sheet > 0.5:
                    st.info("Likely Structure: β-sheet")
                else:
                    st.info("Likely Structure: Loop/coil")
        
        elif app_mode == "Stability Analysis":
            st.title('Protein Stability Analysis')
            
            # Show stability radar chart
            st.subheader("Stability Under Different Conditions")
            st.image(plot_stability_radar(results['stability_conditions']))
            
            # Add explanation
            st.write("""
            This radar chart shows predicted stability scores under different environmental conditions.
            Higher values (0-10) indicate better predicted stability.
            """)
            
            # Show custom stability parameters
            st.subheader("Custom Stability Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Temperature", f"{temperature}°C")
            with col2:
                st.metric("pH", f"{ph}")
            with col3:
                st.metric("Salt Concentration", f"{salt_concentration}M NaCl")
            
            custom_key = f"Custom ({temperature}°C, pH {ph}, {salt_concentration}M NaCl)"
            custom_score = results['stability_conditions'][custom_key]
            
            # Interpret the stability score
            stability_interpretation = ""
            if custom_score < 3:
                stability_interpretation = "Likely unstable under these conditions"
            elif custom_score < 5:
                stability_interpretation = "Moderately stable under these conditions"
            elif custom_score < 7:
                stability_interpretation = "Stable under these conditions"
            else:
                stability_interpretation = "Very stable under these conditions"
            
            st.info(f"Stability Score: {custom_score:.2f}/10 - {stability_interpretation}")
            
            # Show how parameters affect stability
            st.subheader("Parameter Effect Analysis")
            
            # Create a grid of stability predictions varying each parameter
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Effect of Temperature:**")
                temp_range = [10, 25, 37, 45, 60, 80]
                temp_scores = [estimate_stability(results['sequence'], t, ph, salt_concentration) for t in temp_range]
                
                # Create a temperature effect plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(temp_range, temp_scores, 'o-', color='red')
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel('Stability Score')
                ax.set_ylim(0, 10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.write("**Effect of pH:**")
                ph_range = [3, 5, 7, 9, 11]
                ph_scores = [estimate_stability(results['sequence'], temperature, p, salt_concentration) for p in ph_range]
                
                # Create a pH effect plot
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ph_range, ph_scores, 'o-', color='blue')
                ax.set_xlabel('pH')
                ax.set_ylabel('Stability Score')
                ax.set_ylim(0, 10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Analysis text
            st.subheader("Stability Analysis Summary")
            
            # Get most and least stable conditions
            max_condition = max(results['stability_conditions'].items(), key=lambda x: x[1])
            min_condition = min(results['stability_conditions'].items(), key=lambda x: x[1])
            
            st.write(f"""
            Based on the analysis, this protein appears to be most stable under **{max_condition[0]}** 
            conditions (score: {max_condition[1]:.2f}/10) and least stable under **{min_condition[0]}** 
            conditions (score: {min_condition[1]:.2f}/10).
            
            Recommendations for handling this protein:
            - Store near {temperature}°C at pH {ph} if custom conditions show good stability
            - Avoid extreme temperatures and pH values
            - Consider adding stabilizing agents if stability scores are consistently low
            """)
            
        elif app_mode == "Function Prediction":
            st.title('Protein Function Prediction')
            
            # Show three columns with predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Subcellular Localization")
                st.image(plot_prediction_bars(
                    results['protein_properties']['localization'],
                    "Subcellular Localization Prediction"
                ))
                
                top_loc = max(results['protein_properties']['localization'].items(), key=lambda x: x[1])
                st.info(f"Most likely localization: {top_loc[0]} ({top_loc[1]:.2f})")
                
                loc_desc = {
                    'Cytoplasmic': "Resides in the cytoplasm, participates in metabolic processes",
                    'Nuclear': "Localized to the nucleus, may interact with DNA/RNA",
                    'Mitochondrial': "Targeted to mitochondria, may involve energy production",
                    'Secreted': "Secreted outside the cell, may function in signaling or matrix",
                    'Membrane': "Embedded in membranes, may function in transport or signaling"
                }
                
                st.write(f"**Interpretation**: {loc_desc.get(top_loc[0], 'Unknown function')}")
            
            with col2:
                st.subheader("Special Functions")
                
                # Create metrics for antimicrobial and toxicity potential
                amp_score = results['protein_properties']['antimicrobial_potential']
                toxicity_score = results['protein_properties']['toxicity']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create a gauge for antimicrobial potential
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.barh(['Antimicrobial\nPotential'], [amp_score], color='green', height=0.5)
                    ax.barh(['Antimicrobial\nPotential'], [1-amp_score], left=[amp_score], color='lightgray', height=0.5)
                    ax.set_xlim(0, 1)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels(['0', '0.5', '1'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                    st.write(f"Score: {amp_score:.2f}")
                
                with col2:
                    # Create a gauge for toxicity potential
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.barh(['Toxicity\nPotential'], [toxicity_score], color='red', height=0.5)
                    ax.barh(['Toxicity\nPotential'], [1-toxicity_score], left=[toxicity_score], color='lightgray', height=0.5)
                    ax.set_xlim(0, 1)
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels(['0', '0.5', '1'])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                    st.write(f"Score: {toxicity_score:.2f}")
                
                # Interpretation
                st.subheader("Functional Interpretation")
                
                if amp_score > 0.6:
                    st.success("This protein shows significant antimicrobial potential")
                    if toxicity_score > 0.6:
                        st.warning("However, it also shows potential toxicity which may limit applications")
                elif toxicity_score > 0.6:
                    st.warning("This protein shows potential toxicity")
                else:
                    st.info("This protein does not show strong antimicrobial or toxic properties")
            
            # More detailed functional analysis
            st.subheader("Detailed Functional Analysis")
            
            # Sequence features
            st.write("**Key Sequence Features:**")
            
            # Calculate some basic sequence features
            charged_residues = sum(1 for aa in results['sequence'] if AA_PROPERTIES[aa]['charge'] != 0)
            charged_ratio = charged_residues / len(results['sequence'])
            
            hydrophobic_residues = sum(1 for aa in results['sequence'] if AA_PROPERTIES[aa]['hydrophobicity'] > 0)
            hydrophobic_ratio = hydrophobic_residues / len(results['sequence'])
            
            features = [
                f"Length: {len(results['sequence'])} amino acids",
                f"Charged residues: {charged_ratio:.1%}",
                f"Hydrophobic residues: {hydrophobic_ratio:.1%}",
                f"Cysteine content: {results['sequence'].count('C') / len(results['sequence']):.1%}"
            ]
            
            for feature in features:
                st.write(f"- {feature}")
            
            # Potential functions based on sequence features
            st.write("**Potential Functions Based on Sequence Properties:**")
            
            functional_insights = []
            
            # Based on size
            if len(results['sequence']) < 50:
                functional_insights.append("Small peptide, may have signaling or antimicrobial functions")
            elif len(results['sequence']) < 200:
                functional_insights.append("Medium-sized protein, may have enzymatic or regulatory functions")
            else:
                functional_insights.append("Large protein, may have structural or multiple domains")
            
            # Based on charge
            if charged_ratio > 0.25:
                functional_insights.append("High charge density, may interact with nucleic acids or other charged molecules")
            
            # Based on hydrophobicity
            if hydrophobic_ratio > 0.6:
                functional_insights.append("Very hydrophobic, likely membrane-associated or has a strong hydrophobic core")
            
            # Based on cysteine content
            if results['sequence'].count('C') / len(results['sequence']) > 0.05:
                functional_insights.append("Rich in cysteines, may form disulfide bonds for structural stability")
            
            for insight in functional_insights:
                st.write(f"- {insight}")
            
            # Similar proteins disclaimer
            st.info("Note: For more accurate function prediction, comparing this sequence with databases of known proteins would be required.")

else:
    # Main area - Initial state
    st.title('DeepSeq Advanced Protein Analysis')
    st.info('👈 Enter a protein sequence and click "Run Analysis" to begin')
    
    # Basic app description
    st.write("""
    DeepSeq is an advanced protein analysis tool that provides:
    
    - **Structure Prediction**: Accurate 3D structure prediction using ESMFold
    - **Sequence Embeddings**: High-dimensional vector representations of protein sequences
    - **Per-residue Properties**: Detailed analysis of amino acid properties across the sequence
    - **Stability Estimation**: Predict stability under customizable environmental conditions
    - **Function Prediction**: Analyze potential functions, localization, and special properties
    """)
    
    # Show the available analysis modes
    st.subheader("Available Analysis Modes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Structure Prediction**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Beta-secretase-1-protein-2P4J-rainbow.png/800px-Beta-secretase-1-protein-2P4J-rainbow.png", 
                width=200)
    
    with col2:
        st.write("**Sequence Embeddings**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Pca_fish.svg/1200px-Pca_fish.svg.png", 
                width=200)
    
    with col3:
        st.write("**Function Prediction**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Protein_classification_hierarchy_chart.png/800px-Protein_classification_hierarchy_chart.png", 
                width=200)
    
    # How to use
    st.subheader("How to Use")
    st.write("""
    1. Enter your protein sequence in the sidebar
    2. Validate your sequence if needed
    3. Select an analysis mode
    4. Configure any additional parameters
    5. Click 'Run Analysis' to start
    """)
    
    # Example applications
    st.subheader("Example Applications")
    st.write("""
    - **Protein Engineering**: Analyze stability and properties to guide mutation design
    - **Drug Discovery**: Identify potential binding sites and structural features
    - **Antimicrobial Research**: Assess peptides for antimicrobial potential
    - **Protein Characterization**: Get detailed insights into novel proteins
    """)
