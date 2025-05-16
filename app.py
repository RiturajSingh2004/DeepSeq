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

# Configure page
st.set_page_config(page_title="ESMFold Protein Structure Prediction", layout='wide')

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
st.sidebar.write('[*DeepSeq*](https://esmatlas.com/about) is an end-to-end single sequence protein structure predictor based on the ESM-2 language model.')

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

# Protein sequence input
txt = st.sidebar.text_area('Input sequence', DEFAULT_SEQ, height=275)

# Sequence validation dropdown
st.sidebar.subheader("Sequence Validation")
validate_btn = st.sidebar.button('Validate Sequence')

if validate_btn:
    is_valid, invalid_char = validate_sequence(txt)
    if is_valid:
        st.sidebar.success("✅ Sequence contains only valid amino acid codes")
    else:
        st.sidebar.error(f"❌ Invalid character '{invalid_char}' found in sequence. Valid amino acids are: {', '.join(VALID_AMINO_ACIDS)}")

# Visualization options
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

# ESMfold prediction function
def update(sequence=txt):
    # Validate sequence before sending to API
    is_valid, invalid_char = validate_sequence(sequence)
    if not is_valid:
        st.error(f"Invalid character '{invalid_char}' found in sequence. Please fix before predicting.")
        return
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Sending sequence to ESMFold API...")
        progress_bar.progress(10)
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        
        # Simulate API progress for better user experience
        status_text.text("Processing sequence...")
        progress_bar.progress(30)
        time.sleep(0.5)
        
        # Make the API request with error handling and timeout
        try:
            response = requests.post(
                'https://api.esmatlas.com/foldSequence/v1/pdb/', 
                headers=headers, 
                data=sequence,
                timeout=120  # 2 minute timeout
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {str(e)}")
            progress_bar.empty()
            status_text.empty()
            return
        
        status_text.text("Processing structure data...")
        progress_bar.progress(70)
        
        pdb_string = response.content.decode('utf-8')
        
        # Save PDB locally
        with open('predicted.pdb', 'w') as f:
            f.write(pdb_string)
        
        # Calculate plDDT value
        try:
            struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
            b_value = round(struct.b_factor.mean(), 4)
        except Exception as e:
            st.error(f"Error processing PDB data: {str(e)}")
            b_value = "Error calculating plDDT"
        
        progress_bar.progress(90)
        status_text.text("Rendering visualization...")
        
        # Main area
        st.title('DeepSeq Protein Structure Prediction')

        # Display protein structure
        st.subheader('Visualization of predicted protein structure')
        render_mol(
            pdb_string, 
            style=viz_style, 
            color_scheme=color_scheme, 
            spin=spin_model,
            background=background_color
        )
        
        # plDDT value display
        st.subheader('plDDT')
        st.write('plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.')
        st.info(f'plDDT: {b_value}')
        
        # Download button
        st.download_button(
            label="Download PDB",
            data=pdb_string,
            file_name='predicted.pdb',
            mime='text/plain',
        )
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("Prediction complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()



# Predict button
predict = st.sidebar.button('Predict', on_click=update)

# Initial state
if not predict:
    # Main area
    st.title('DeepSeq Protein Structure Prediction')
    st.info('👈 Enter a protein sequence and click "Predict" to visualize its structure')
    
    # Basic app instructions
    st.subheader("How to use this app")
    st.write("""
    1. Enter a protein sequence in the sidebar text area
    2. Optionally validate your sequence by clicking "Validate Sequence"
    3. Select visualization options for the resulting structure
    4. Click "Predict" to get the 3D structure prediction
    5. Download the PDB file for further analysis
    """)
    
    # Show example of what will be displayed
    st.subheader("Example Output")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Beta-secretase-1-protein-2P4J-rainbow.png/800px-Beta-secretase-1-protein-2P4J-rainbow.png", 
             caption="Example protein structure visualization (not from current sequence)", width=400)
