import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
import io

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
st.set_page_config(
    page_title="Radix Thread Abrasion Graph",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to enforce dark mode and elegant styling
st.markdown("""
<style>
    /* Ensure the entire app has a dark background and consistent typography */
    .stApp {
        background-color: #0E1117; /* Dark background */
        color: #F0F2F6; /* Light text color */
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Main Background & Structure */
    .main {
        background-color: #0E1117;
        color: #F0F2F6;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    h1 {
        font-size: 36px;
        padding-bottom: 10px;
        border-bottom: 2px solid #64CCC9; /* Teal accent for header underline */
    }
    h2, h3 {
        font-weight: 600;
    }
    p, li, span, blockquote {
        color: #F0F2F6;
    }
    
    /* Links */
    a {
        color: #64CCC9; /* Teal for links */
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #64CCC9; /* Teal button background */
        color: #0E1117; /* Dark text on button */
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #54B8B5; /* Slightly darker teal on hover */
        box-shadow: 0 5px 15px rgba(100, 204, 201, 0.3); /* Soft shadow */
        transform: translateY(-2px); /* Slight lift effect */
    }
    
    /* Generic Streamlit Containers (e.g., for columns) */
    .css-1v3fvcr { /* Specific class for some Streamlit containers */
        background-color: #0E1117;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0; /* Grey for inactive tabs */
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 8px;
        background-color: #1C1F26; /* Darker background for tab container */
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px;
        border-radius: 8px;
        background-color: #1C1F26; /* Darker background for tab content */
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #64CCC9; /* Teal highlight for active tab */
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        background-color: #1C1F26;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
        color: #F0F2F6;
    }
    .stDataFrame th {
        background-color: #2C303A !important; /* Darker header background */
        color: #64CCC9 !important; /* Teal header text */
        font-weight: 600;
    }
    .stDataFrame td {
        background-color: #1C1F26 !important; /* Darker cell background */
        color: #F0F2F6 !important; /* Light cell text */
        border-bottom: 1px solid #2C303A !important; /* Subtle border */
    }
    
    /* Hide Streamlit Footer and Main Menu */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    
    /* Alert Messages */
    .stAlert {
        border-radius: 8px;
        background-color: #1C1F26;
        color: #F0F2F6;
        border-left: 4px solid #64CCC9; /* Teal border for info/general alerts */
    }
    div[data-baseweb="notification"] { /* For success/error toasts */
        background-color: #1C1F26 !important;
        color: #F0F2F6 !important;
    }
    .success {
        border-left: 4px solid #28A745 !important; /* Green for success */
    }
    .error {
        border-left: 4px solid #DC3545 !important; /* Red for error */
    }
    
    /* General Container Styling */
    .css-1r6slb0 { /* Another common Streamlit container class */
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        padding: 20px;
        background-color: #1C1F26 !important;
    }
    
    /* Custom Card for Results/Important Info */
    .dark-card {
        background-color: #1C1F26;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.5); /* Stronger shadow */
        margin-bottom: 25px;
        text-align: center;
    }
    
    /* Header Styling */
    .app-header {
        background-color: #1C1F26;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .radix-logo {
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        font-size: 38px;
        color: #64CCC9; /* Teal for logo */
        margin-bottom: 5px;
        letter-spacing: 2px;
        text-shadow: 0 2px 10px rgba(100, 204, 201, 0.3);
    }
    
    /* Data Editor */
    [data-testid="stDataEditor"] {
        background-color: #1C1F26 !important;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2C303A; /* Subtle border */
    }
    [data-testid="stDataEditor"] .cell {
        background-color: #1C1F26 !important;
        color: #F0F2F6 !important;
    }
    [data-testid="stDataEditor"] .cell:focus {
        background-color: #2C303A !important;
        border: 1px solid #64CCC9 !important;
    }
    [data-testid="stDataEditor"] .header {
        background-color: #2C303A !important;
        color: #64CCC9 !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #1C1F26 !important;
        border-radius: 8px;
        padding: 15px !important;
        border: 1px dashed #64CCC9 !important; /* Dashed teal border */
    }
    [data-testid="stFileUploader"] p {
        color: #A0A0A0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2C303A !important;
        margin: 25px 0 !important;
    }
    
    /* Selection Color */
    ::selection {
        background-color: rgba(100, 204, 201, 0.3);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1C1F26 !important;
        color: #F0F2F6 !important;
        border-radius: 8px !important;
        border: 1px solid #2C303A;
    }
    .streamlit-expanderContent {
        background-color: #1C1F26 !important;
        border-radius: 0 0 8px 8px !important;
        border-top: 1px solid #2C303A;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #2C303A;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #64CCC9;
    }
</style>
""", unsafe_allow_html=True)

# Header Aplikasi with Radix Logo
st.markdown("""
<div class="app-header">
    <div class="radix-logo">RADIX</div>
    <h1 style="margin-top: 0; color: #FFFFFF; font-size: 36px;">Analisis Abrasi Benang</h1>
    <p style="color: #A0A0A0; font-size: 16px;">Alat profesional untuk memvisualisasikan data abrasi benang dan menghitung nilai persimpangan</p>
</div>
""", unsafe_allow_html=True)

# Define initial data
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# Initialize session state for data if not present
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)

if 'update_graph' not in st.session_state:
    st.session_state.update_graph = False

# Create tabs for different data input methods
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.subheader("Input Data Manual")
    st.write("Modifikasi nilai sumbu-y (N atau nilai benang putus):")
    
    # Create editable dataframe for y-values
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
        on_change=None # Don't auto-update
    )
    
    # Add buttons to apply changes
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes"):
            st.session_state.data['y_values'] = edited_df['y_value'].tolist()
            st.session_state.update_graph = True
            st.success("Data berhasil diperbarui!")
    
    with col2:
        # Button to reset to initial values
        if st.button("Reset ke Nilai Awal", key="reset_values"):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.update_graph = True
            st.success("Data direset ke nilai awal!")

with tabs[1]:
    st.subheader("Impor Data dari Excel")
    st.write("Unggah file Excel dengan nilai x dan y (harus memiliki kolom 'x_values' dan 'y_values')")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            # Check if required columns exist
            if 'x_values' in df.columns and 'y_values' in df.columns:
                # Preview data
                st.write("Pratinjau Data:")
                st.dataframe(df.head(), use_container_width=True)
                
                # Button to apply imported data
                if st.button("Gunakan Data Ini", key="use_imported"):
                    st.session_state.data = df[['x_values', 'y_values']]
                    st.session_state.update_graph = True
                    st.success("Data yang diimpor berhasil diterapkan!")
            else:
                st.error("File Excel harus berisi kolom 'x_values' dan 'y_values'.")
        except Exception as e:
            st.error(f"Error membaca file Excel: {e}")

# Display example Excel template for download
st.divider()
if st.button("Unduh Template Excel Contoh"):
    # Create example DataFrame
    sample_df = pd.DataFrame(INITIAL_DATA)
    
    # Create a buffer to save the Excel file
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False)
    buffer.seek(0)
    
    # Create download button
    st.download_button(
        label="Klik untuk Mengunduh",
        data=buffer,
        file_name="thread_abrasion_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Create graph section
st.divider()
st.subheader("Grafik Abrasi Benang")

# Plot graph
if st.session_state.update_graph or not 'fig' in st.session_state:
    # Get data from session state
    x_values = st.session_state.data['x_values']
    y_values = st.session_state.data['y_values']
    
    # Create interpolation function
    f = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    
    # Find indices for 10th and 20th points in the dataset
    x_values_list = list(x_values)
    
    # Find 10th point (approximately the 10th value in the dataset)
    point_10_index = min(9, len(x_values_list) - 1) # Ensure index is within bounds
    specific_x1 = x_values_list[point_10_index]
    specific_y1 = y_values[point_10_index]
    
    # Find 20th point (approximately the 20th value in the dataset)
    point_20_index = min(19, len(x_values_list) - 1) # Ensure index is within bounds
    specific_x2 = x_values_list[point_20_index]
    specific_y2 = y_values[point_20_index]
    
    # Calculate slope and intercept of the line connecting these two points
    if (specific_x2 - specific_x1) != 0: # Avoid division by zero
        slope = (specific_y2 - specific_y1) / (specific_x2 - specific_x1)
        intercept = specific_y1 - slope * specific_x1
    else: # If x1 and x2 are the same, it's a vertical line or error case
        slope = 0
        intercept = specific_y1 # Treat as horizontal at y1 for plotting line projection
        st.warning("Titik ke-10 dan ke-20 memiliki nilai X yang sama. Proyeksi garis mungkin tidak akurat.")

    # Function to calculate y value using the line equation
    def line_equation(x):
        return slope * x + intercept
    
    # Calculate y value at x=50 using the line equation
    y_at_x_50_line = line_equation(50)
    
    # Interpolate original curve at specific points
    y_at_x_10 = float(f(10))
    y_at_x_20 = float(f(20))
    y_at_x_50 = float(f(50))
    
    # Create figure
    fig = go.Figure()
    
    # Add main data line
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values,
        mode='lines+markers',
        name='Data Abrasi',
        line=dict(color='#64CCC9', width=3), # Teal color for main line
        marker=dict(size=8, color='#64CCC9')
    ))
    
    # Add specific points (10th and 20th points)
    fig.add_trace(go.Scatter(
        x=[specific_x1, specific_x2],
        y=[specific_y1, specific_y2],
        mode='markers',
        name='Titik Referensi Terpilih',
        marker=dict(size=12, color='#FFD700', symbol='star', line=dict(width=2, color='white')) # Gold star
    ))
    
    # Add line connecting the two specific points and extending to x=50
    fig.add_shape(
        type="line",
        x0=specific_x1,
        y0=specific_y1,
        x1=specific_x2,
        y1=specific_y2,
        line=dict(color="#FF6347", width=3), # Tomato red for the line
    )
    
    # Extend line to x=50
    fig.add_shape(
        type="line",
        x0=specific_x2,
        y0=specific_y2,
        x1=50,
        y1=y_at_x_50_line,
        line=dict(color="#FF6347", width=3, dash="dot"), # Dotted extension
    )
    
    # Add point at the line's intersection with x=50
    fig.add_trace(go.Scatter(
        x=[50],
        y=[y_at_x_50_line],
        mode='markers',
        name=f'Persimpangan Garis pada x=50, y={y_at_x_50_line:.2f}',
        marker=dict(size=14, color='#FF6347', symbol='circle-open', line=dict(width=3, color='#FF6347'))
    ))
    
    # Add label for linear extrapolation at x=50
    fig.add_annotation(
        x=50,
        y=y_at_x_50_line + 50, # Offset to avoid overlap
        text=f"Nilai Garis: {y_at_x_50_line:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#FF6347',
        font=dict(size=14, color='#FF6347', family="Arial, sans-serif"),
    )
    
    # Add vertical lines at x=17.7 (10th point) and x=40.4 (20th point)
    for x_pos, y_val, color, name, index in [
        (specific_x1, specific_y1, "#FFD700", f"Titik 10 ({specific_x1}, {specific_y1})", 10),
        (specific_x2, specific_y2, "#FFD700", f"Titik 20 ({specific_x2}, {specific_y2})", 20),
    ]:
        # Add vertical dashed line
        fig.add_shape(
            type="line",
            x0=x_pos,
            y0=0,
            x1=x_pos,
            y1=max(y_values) * 1.1, # Extend beyond max y for visibility
            line=dict(color=color, width=1.5, dash="dash"),
        )
        
        # Add text label for vertical line
        fig.add_annotation(
            x=x_pos,
            y=y_val - 50 if y_val > 50 else y_val + 50, # Offset below point, adjust if too low
            text=f"Titik {index}",
            showarrow=False,
            font=dict(color=color, size=12, family="Arial, sans-serif", weight="bold")
        )
    
    # Add vertical line at x=50
    fig.add_shape(
        type="line",
        x0=50,
        y0=0,
        x1=50,
        y1=max(y_values) * 1.1,
        line=dict(color="#DC3545", width=2, dash="dash"), # Red for x=50
    )
    
    # Add text label for x=50 line
    fig.add_annotation(
        x=50,
        y=max(y_values) * 1.05,
        text="x=50",
        showarrow=False,
        font=dict(color="#DC3545", size=14, family="Arial, sans-serif", weight="bold")
    )
    
    # Add point at the intersection of the original curve with x=50
    fig.add_trace(go.Scatter(
        x=[50],
        y=[y_at_x_50],
        mode='markers',
        name=f'Persimpangan pada x=50, y={y_at_x_50:.2f}',
        marker=dict(size=14, color='#DC3545', symbol='circle', line=dict(width=2, color='white')) # Solid red circle
    ))
    
    # Update layout with more elegant dark theme styling for plotly
    fig.update_layout(
        title=None, # Title is already in markdown
        xaxis_title=dict(text="Nilai yang sudah tetap", font=dict(family="Arial, sans-serif", size=14, color="#F0F2F6")),
        yaxis_title=dict(text="N atau nilai benang putus", font=dict(family="Arial, sans-serif", size=14, color="#F0F2F6")),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            font=dict(family="Arial, sans-serif", size=12, color="#F0F2F6"),
            bgcolor="rgba(28,31,38,0.7)", # Semi-transparent dark background for legend
            bordercolor="#2C303A",
            borderwidth=1,
            itemclick="toggleothers" # Allow clicking legend items to toggle visibility
        ),
        margin=dict(l=40, r=40, t=20, b=40), # More margin
        height=600,
        plot_bgcolor="#1C1F26", # Dark background for plot area
        paper_bgcolor="#1C1F26", # Dark background for entire figure
        font=dict(family="Arial, sans-serif", size=12, color="#F0F2F6"),
        hovermode="x unified" # Unified hover tooltips
    )
    
    # Add gridlines for better readability
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#2C303A', # Darker grid lines
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='#2C303A',
        showline=True,
        linewidth=1.5,
        linecolor='#2C303A',
        tickfont=dict(color="#A0A0A0")
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#2C303A',
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='#2C303A',
        showline=True,
        linewidth=1.5,
        linecolor='#2C303A',
        tickfont=dict(color="#A0A0A0")
    )
    
    # Save in session state
    st.session_state.fig = fig
    st.session_state.y_at_x_50 = y_at_x_50
    st.session_state.update_graph = False

# Display the graph
st.plotly_chart(st.session_state.fig, use_container_width=True)

# Function to recalculate line values using session state data
def recalculate_line_values():
    x_values = st.session_state.data['x_values']
    y_values = st.session_state.data['y_values']
    
    x_values_list = list(x_values)
    
    point_10_index = min(9, len(x_values_list) - 1)
    specific_x1 = x_values_list[point_10_index]
    specific_y1 = y_values[point_10_index]
    
    point_20_index = min(19, len(x_values_list) - 1)
    specific_x2 = x_values_list[point_20_index]
    specific_y2 = y_values[point_20_index]
    
    if (specific_x2 - specific_x1) != 0:
        slope = (specific_y2 - specific_y1) / (specific_x2 - specific_x1)
        intercept = specific_y1 - slope * specific_x1
    else:
        slope = 0
        intercept = specific_y1
    
    return slope * 50 + intercept, specific_x1, specific_y1, specific_x2, specific_y2

# Calculate line intersection and get points
y_line_intersection, p10_x, p10_y, p20_x, p20_y = recalculate_line_values()

# Display results in a cleaner format
st.markdown(f"""
<div class="dark-card">
    <h2 style="color: #FFFFFF; margin-bottom: 5px;">Hasil Analisis pada x=50</h2>
    <h1 style="color: #64CCC9; font-size: 48px; margin: 10px 0;">{y_line_intersection:.2f}</h1>
    <p style="color: #A0A0A0; font-size: 16px;">Nilai perpotongan garis pada x=50</p>
    <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
        Berdasarkan garis antara titik 10 ({p10_x:.2f}, {p10_y:.2f}) dan titik 20 ({p20_x:.2f}, {p20_y:.2f})
    </div>
</div>
""", unsafe_allow_html=True)

# Display table data view
with st.expander("Lihat Tabel Data Lengkap"):
    st.dataframe(
        pd.DataFrame({
            'x_values (Nilai yang sudah tetap)': st.session_state.data['x_values'],
            'y_values (N atau nilai benang putus)': st.session_state.data['y_values']
        }),
        hide_index=True,
        use_container_width=True
    )

# Add important graph information
st.markdown("""
<div class="dark-card">
    <h3 style="color: #FFFFFF;">Analisis Abrasi Benang</h3>
    <p style="color: #F0F2F6;">Grafik menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y).</p>
    <ul style="color: #F0F2F6;">
        <li><strong style="color: #64CCC9;">Garis teal</strong> adalah kurva data abrasi benang.</li>
        <li><strong style="color: #FFD700;">Bintang emas</strong> menandai titik ke-10 dan ke-20 dari dataset.</li>
        <li><strong style="color: #FF6347;">Garis oranye/merah</strong> menghubungkan kedua titik tersebut dan diperpanjang ke x=50.</li>
        <li><strong style="color: #FF6347;">Lingkaran kosong</strong> menunjukkan perpotongan garis dengan x=50.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Add additional information in an expander
with st.expander("Informasi Tambahan & Tips Interaksi"):
    st.markdown("""
    <div style="padding: 10px;">
        <h4 style="color: #FFFFFF;">Poin Penting:</h4>
        <ul style="color: #F0F2F6;">
            <li>Garis yang ditarik antara titik ke-10 dan ke-20 memberikan proyeksi linear ke x=50.</li>
            <li>Nilai pada garis saat x=50 dianggap sebagai "hasil" resmi untuk analisis ini.</li>
        </ul>
        
        <h4 style="color: #FFFFFF; margin-top: 20px;">Tips Interaksi:</h4>
        <ul style="color: #F0F2F6;">
            <li>Arahkan kursor ke titik untuk melihat nilainya.</li>
            <li>Klik dan seret untuk memperbesar area tertentu.</li>
            <li>Klik dua kali untuk mengatur ulang tampilan.</li>
            <li>Gunakan toolbar di kanan atas grafik untuk opsi lainnya.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Add Radix footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 15px; font-size: 14px; font-family: 'Helvetica Neue', sans-serif; color: #A0A0A0; border-top: 1px solid #2C303A; background-color: #1C1F26; border-radius: 0 0 8px 8px;">
    <p>Dikembangkan oleh <span style="font-weight: bold; color: #64CCC9;">RADIX</span> &copy; 2025</p>
    <p>Solusi Analisis Abrasi Benang Profesional</p>
</div>
""", unsafe_allow_html=True)
