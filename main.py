import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor, LinearRegression # Import for RANSAC

# --- START OF ACCESS CODE IMPLEMENTATION ---
# Define your secret code
ACCESS_CODE = "RADIX2025" # GANTI INI DENGAN KODE YANG ANDA INGINKAN!

def check_password():
    """Returns True if the user enters the correct password, False otherwise."""
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        st.subheader("Masukkan Kode Akses")
        password_input = st.text_input("Kode Akses", type="password", key="password_input")
        if st.button("Masuk", key="login_button"):
            if password_input == ACCESS_CODE:
                st.session_state.password_entered = True
                st.rerun() # Rerun to hide the login screen
            else:
                st.error("Kode akses salah. Silakan coba lagi.")
        return False
    return True

# Call the password check function at the very beginning of your app
if not check_password():
    st.stop() # Stop execution if password is not entered or is incorrect
# --- END OF ACCESS CODE IMPLEMENTATION ---


# Set page configuration
st.set_set_page_config(
    page_title="Radix Thread Abrasion Graph",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for Dark Mode elegant styling
st.markdown("""
<style>
    /* Main Background & Structure */
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: 0.5px;
    }
    h1 {
        font-weight: 700;
        font-size: 36px;
        padding-bottom: 10px;
        border-bottom: 2px solid #4F8EF7;
    }
    h2, h3 {
        font-weight: 600;
    }
    p, li, span {
        color: #E0E0E0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4F8EF7;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1E90FF;
        box-shadow: 0 5px 15px rgba(79, 142, 247, 0.2);
        transform: translateY(-2px);
    }
    
    /* Containers */
    .css-1v3fvcr { /* This might be an outdated class, check Streamlit's latest class for main container */
        background-color: #121212;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 8px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px;
        border-radius: 8px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4F8EF7;
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        background-color: #1E1E1E;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
    }
    .stDataFrame th {
        background-color: #2E2E2E !important;
        color: #4F8EF7 !important;
        font-weight: 600;
    }
    .stDataFrame td {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border-bottom: 1px solid #2E2E2E !important;
    }
    
    /* Footer & Hidden Elements */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
        background-color: #1E1E1E;
    }
    div[data-baseweb="notification"] {
        background-color: #1E1E1E !important;
    }
    
    /* Additional Styling */
    .css-1r6slb0 { /* Target the main block container for expander, might need update */
        background-color: #1E1E1E !important;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        padding: 20px;
    }
     .st-emotion-cache-1wv8c4k { /* This class seems to target the expander header and content */
        background-color: #1E1E1E !important;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        padding: 10px;
    }
    .st-emotion-cache-1wv8c4k div[data-testid="stExpanderToggle"] {
        background-color: #1E1E1E; /* Ensure expander header is dark */
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #121212;
    }
    ::-webkit-scrollbar-thumb {
        background: #2E2E2E;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4F8EF7;
    }
    
    /* Success/Error Messages */
    .success {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border-left: 4px solid #28A745 !important;
    }
    .error {
        background-color: rgba(220, 53, 69, 0.1) !important;
        border-left: 4px solid #DC3545 !important;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    /* Radix Header/Footer */
    .radix-logo {
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        font-size: 38px;
        color: #4F8EF7;
        margin-bottom: 5px;
        letter-spacing: 2px;
        text-shadow: 0 2px 10px rgba(79, 142, 247, 0.3);
    }
    .app-header {
        background-color: #1E1E1E;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(5px);
    }
    .radix-footer {
        text-align: center;
        margin-top: 40px;
        padding: 15px;
        font-size: 14px;
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0;
        border-top: 1px solid #2E2E2E;
        background-color: #1E1E1E;
        border-radius: 0 0 8px 8px;
    }

    /* Info Alert */
    .stInfo {
        background-color: rgba(79, 142, 247, 0.1);
        border-radius: 8px;
        padding: 15px;
        color: #FFFFFF;
        border-left: 4px solid #4F8EF7;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        font-weight: 500;
    }
    
    /* Data Editor specific styling */
    [data-testid="stDataEditor"] {
        background-color: #1E1E1E !important;
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stDataEditor"] .cell {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
    }
    [data-testid="stDataEditor"] .cell:focus {
        background-color: #2E2E2E !important;
        border: 1px solid #4F8EF7 !important;
    }
    [data-testid="stDataEditor"] .header {
        background-color: #2E2E2E !important;
        color: #4F8EF7 !important;
    }
    
    /* File Uploader specific styling */
    [data-testid="stFileUploader"] {
        background-color: #1E1E1E !important;
        border-radius: 8px;
        padding: 15px !important;
        border: 1px dashed #4F8EF7 !important;
    }
    [data-testid="stFileUploader"] p {
        color: #A0A0A0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2E2E2E !important;
        margin: 25px 0 !important;
    }
    
    /* Selection Color */
    ::selection {
        background-color: rgba(79, 142, 247, 0.3);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: #1E1E1E !important;
        border-radius: 0 0 8px 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header with Radix Logo
st.markdown("""
<div class="app-header">
    <div class="radix-logo">RADIX</div>
    <h1 style="margin-top: 0; color: #FFFFFF; font-size: 36px;">Thread Abrasion Analysis</h1>
    <p style="color: #A0A0A0; font-size: 16px;">A professional tool for visualizing thread abrasion data and calculating intersection values</p>
</div>
""", unsafe_allow_html=True)

# Define initial data
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# Initialize session state for data if not exists
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)

if 'update_graph' not in st.session_state:
    st.session_state.update_graph = False

# Create tabs for different ways to input data
tabs = st.tabs(["Manual Input", "Import from Excel"])

with tabs[0]:
    st.subheader("Manual Data Input")
    st.write("Modify y-axis values (N atau nilai benang putus):")
    
    # Create an editable dataframe for the y-values
    edited_data = pd.DataFrame({
        'x_value': st.session_state.data['x_values'],
        'y_value': st.session_state.data['y_values']
    })
    
    # Make it editable
    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=True,
        use_container_width=True,
        key="data_editor",
        on_change=None # Don't update automatically
    )
