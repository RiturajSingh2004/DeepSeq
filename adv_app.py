# This app uses py3Dmol directly for protein visualization
# With enhanced coloring and visualization options

import streamlit as st
import requests
import re
import time
import os
import biotite.structure.io as bsio

# Configure page
st.set_page_config(page_title="ESMFold Protein Structure Prediction", layout='wide')

# Sidebar
st.sidebar.title('🎈 ESMFold')
st.sidebar.write('[*ESMFold*](https://esmatlas.com/about) is an end-to-end single sequence protein structure predictor based on the ESM-2 language model. For more information, read the [research article](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v2) and the [news article](https://www.nature.com/articles/d41586-022-03539-1) published in *Nature*.')

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

# Function to render protein structure with py3Dmol
def render_mol_html(pdb_string, style='cartoon', color_scheme='spectrum', spin=True, background='white'):
    """Create a py3Dmol visualization HTML"""
    
    # Create a viewer div with style
    viewer_html = f"""
    <div id="3dmol" style="height: 500px; width: 100%; position: relative;">
    <p id="loadingMsg" style="background-color: #ffcc00; padding: 8px; border-radius: 4px; position: absolute; 
                              top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 10;">
    Loading protein structure...</p>
    </div>
    <script>
    $(document).ready(function() {{
        let viewer = $3Dmol.createViewer('3dmol');
        let v = viewer;
        v.addModel(`{pdb_string}`, "pdb");
        
        // Style setting
        if ('{style}' === 'cartoon') {{
            v.setStyle({{cartoon: {{}}}}); 
            // Apply coloring
            if ('{color_scheme}' === 'spectrum') {{
                v.setStyle({{cartoon: {{color: 'spectrum'}}}}); 
            }} else if ('{color_scheme}' === 'chainbow') {{
                v.setStyle({{cartoon: {{color: 'chainbow'}}}}); 
            }} else if ('{color_scheme}' === 'ssPyMol') {{
                v.setStyle({{cartoon: {{color: 'ssPyMol', colorScheme: 'pyMol'}}}}); 
            }} else if ('{color_scheme}' === 'rainbow') {{
                v.setStyle({{cartoon: {{colorfunc: color => color.r = (Math.sin(color.atom.resi*0.5)+1)/2, colorScheme: {{prop: 'resi', gradient: 'roygb'}}}}}}); 
            }} else if ('{color_scheme}' === 'structure') {{
                v.setStyle({{'sheet': {{cartoon: {{color: '#ffd42a'}}}}}}); 
                v.setStyle({{'helix': {{cartoon: {{color: '#0097cc'}}}}}}); 
                v.setStyle({{'noss': {{cartoon: {{color: '#ffffff'}}}}}}); 
            }}
        }} else if ('{style}' === 'stick') {{
            v.setStyle({{stick: {{}}}}); 
            if ('{color_scheme}' === 'spectrum') {{
                v.setStyle({{stick: {{colorscheme: 'yellowBlue'}}}}); 
            }} else if ('{color_scheme}' === 'element') {{
                v.setStyle({{stick: {{colorscheme: 'default'}}}}); 
            }}
        }} else if ('{style}' === 'sphere') {{
            v.setStyle({{sphere: {{}}}}); 
            if ('{color_scheme}' === 'spectrum') {{
                v.setStyle({{sphere: {{colorscheme: 'yellowBlue'}}}}); 
            }} else if ('{color_scheme}' === 'element') {{
                v.setStyle({{sphere: {{colorscheme: 'default'}}}}); 
            }}
        }} else if ('{style}' === 'line') {{
            v.setStyle({{line: {{color: 'spectrum'}}}}); 
        }} else if ('{style}' === 'cross') {{
            v.setStyle({{cross: {{color: 'spectrum'}}}}); 
        }} else if ('{style}' === 'surface') {{
            v.addSurface($3Dmol.SurfaceType.VDW, {{opacity:0.8, color:'{color_scheme}'}});
        }} else if ('{style}' === 'cartoon_and_stick') {{
            v.setStyle({{cartoon: {{color: 'spectrum'}}}}); 
            v.addStyle({{stick: {{radius: 0.2, colorscheme: 'default'}}}});
        }}
        
        v.setBackgroundColor('{background}');
        v.zoomTo();
        
        if ({str(spin).lower()}) {{
            v.spin(true);
        }}
        
        v.render();
        
        // Hide loading message when viewer is ready
        $('#loadingMsg').hide();
    }});
    </script>
    """
    return viewer_html

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
    ('cartoon', 'stick', 'sphere', 'surface', 'cartoon_and_stick', 'line', 'cross')
)
color_scheme = st.sidebar.selectbox(
    'Color scheme',
    ('spectrum', 'chainbow', 'rainbow', 'structure', 'element', 'ssPyMol', 'red', 'green', 'blue')
)
background_color = st.sidebar.selectbox(
    'Background color',
    ('white', 'black', 'gray', '#f0f0f0')
)
spin_model = st.sidebar.checkbox('Spin model', value=True)

# Add 3Dmol.js dependencies - direct import from JSDelivr CDN
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/3dmol@1.8.0/build/3Dmol-min.js"></script>
""", unsafe_allow_html=True)

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
        
        # Save PDB locally for analysis
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
        
        # Display protein structure
        st.subheader('Visualization of predicted protein structure')
        
        # Generate HTML for the 3Dmol viewer with our custom function
        viewer_html = render_mol_html(
            pdb_string, 
            style=viz_style, 
            color_scheme=color_scheme, 
            spin=spin_model,
            background=background_color
        )
        
        # Display the customized 3Dmol viewer
        st.components.v1.html(viewer_html, height=520, scrolling=False)
        
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

# Main area
st.title('ESMFold Protein Structure Prediction')

# Predict button
predict = st.sidebar.button('Predict', on_click=update)

# Initial state
if not predict:
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Beta-secretase-1-protein-2P4J-rainbow.png/800px-Beta-secretase-1-protein-2P4J-rainbow.png", 
             caption="Example protein structure visualization (not from current sequence)", width=400)