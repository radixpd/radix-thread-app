import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate, stats # Import stats for linear regression
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
    .css-1v3fvcr { /* Specific class for some Streamlit containers */
        background-color: #0E1117;
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
    .pulcra-logo {
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
    <div class="pulcra-logo">PULCRA</div>
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

# --- Caching untuk Performansi ---
@st.cache_data
def get_initial_data_df():
    return pd.DataFrame(INITIAL_DATA)

@st.cache_data
def load_excel_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'x_values' in df.columns and 'y_values' in df.columns:
                return df[['x_values', 'y_values']]
            else:
                st.error("File Excel harus berisi kolom 'x_values' dan 'y_values'.")
                return None
        except Exception as e:
            st.error(f"Error membaca file Excel: {e}")
            return None
    return None

@st.cache_data
def generate_graph_data(x_values, y_values):
    # Buat fungsi interpolasi dari data asli (untuk kurva teal)
    f_curve = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    
    # Hitung garis regresi linear (best fit) untuk SELURUH dataset
    # Ini akan menjadi "garis oranye" yang melewati paling banyak titik atau mendekati banyak titik
    
    # Pastikan ada cukup data untuk regresi
    if len(x_values) < 2:
        st.warning("Tidak cukup data untuk melakukan regresi linear. Perlu setidaknya 2 titik.")
        # Fallback to a simple line if not enough data
        slope = 0
        intercept = y_values.iloc[0] if not y_values.empty else 0
        if len(x_values) == 1:
            slope = 0 # Horizontal line if only one point
        elif len(x_values) > 1 and (x_values.iloc[-1] - x_values.iloc[0]) != 0:
            slope = (y_values.iloc[-1] - y_values.iloc[0]) / (x_values.iloc[-1] - x_values.iloc[0])
            intercept = y_values.iloc[0] - slope * x_values.iloc[0]

    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)

    # Hitung nilai y pada x=50 menggunakan garis regresi
    y_at_x_50_line = slope * 50 + intercept
    
    # Hitung nilai y pada x=50 menggunakan kurva asli (untuk lingkaran merah)
    y_at_x_50_curve = float(f_curve(50))

    # Kita hanya mengembalikan slope dan intercept dari regresi global
    return slope, intercept, y_at_x_50_line, y_at_x_50_curve

@st.cache_data
def generate_plotly_figure(x_values, y_values, slope, intercept, y_at_x_50_line, y_at_x_50_curve):
    fig = go.Figure()
    
    # Add main data line (teal curve)
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values,
        mode='lines+markers',
        name='Data Abrasi',
        line=dict(color='#64CCC9', width=3),
        marker=dict(size=8, color='#64CCC9')
    ))
    
    # Define x-range for the regression line (from min x to max x, then extrapolate to 50)
    x_line_start = x_values.min()
    x_line_end_initial = x_values.max()
    x_line_extrapolate_to = 50

    # Calculate y-values for the regression line over the relevant x-range
    y_line_start = slope * x_line_start + intercept
    y_line_end_initial = slope * x_line_end_initial + intercept
    
    # Add the regression line (orange line)
    # This line now represents the "best fit" through all points
    fig.add_trace(go.Scatter(
        x=[x_line_start, x_line_end_initial, x_line_extrapolate_to],
        y=[y_line_start, y_line_end_initial, y_at_x_50_line],
        mode='lines',
        name='Garis Regresi Linear (Kecocokan Terbaik)', # Changed name
        line=dict(color="#FF6347", width=3, dash="dot"), # Dotted for emphasis on extrapolation
    ))

    # Add point at the line's intersection with x=50
    fig.add_trace(go.Scatter(
        x=[50],
        y=[y_at_x_50_line],
        mode='markers',
        name=f'Perpotongan Garis Regresi pada x=50, y={y_at_x_50_line:.2f}', # Changed name
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
        y=[y_at_x_50_curve],
        mode='markers',
        name=f'Perpotongan Kurva Asli pada x=50, y={y_at_x_50_curve:.2f}', # Changed name
        marker=dict(size=14, color='#DC3545', symbol='circle', line=dict(width=2, color='white')) # Solid red circle
    ))
    
    # Update layout with more elegant dark theme styling for plotly
    fig.update_layout(
        title=None,
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
            itemclick="toggleothers"
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=600,
        plot_bgcolor="#1C1F26",
        paper_bgcolor="#1C1F26",
        font=dict(family="Arial, sans-serif", size=12, color="#F0F2F6"),
        hovermode="x unified"
    )
    
    # Add gridlines for better readability
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#2C303A', zeroline=True, zerolinewidth=1.5, zerolinecolor='#2C303A', showline=True, linewidth=1.5, linecolor='#2C303A', tickfont=dict(color="#A0A0A0")
    )
    
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#2C303A', zeroline=True, zerolinewidth=1.5, zerolinecolor='#2C303A', showline=True, linewidth=1.5, linecolor='#2C303A', tickfont=dict(color="#A0A0A0")
    )
    return fig

# Create tabs for different data input methods
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.subheader("Input Data Manual")
    st.write("Modifikasi nilai sumbu-y (N atau nilai benang putus):")
    
    edited_data = pd.DataFrame({
        'x_value': st.session_state.data['x_values'],
        'y_value': st.session_state.data['y_values']
    })
    
    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=True,
        use_container_width=True,
        key="data_editor",
        on_change=None # Don't auto-update
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes"):
            st.session_state.data['y_values'] = edited_df['y_value'].tolist()
            # Clear caches that depend on data to force recalculation
            generate_graph_data.clear()
            generate_plotly_figure.clear()
            st.session_state.update_graph = True
            st.success("Data berhasil diperbarui!")
    
    with col2:
        if st.button("Reset ke Nilai Awal", key="reset_values"):
            st.session_state.data = get_initial_data_df() # Use cached function
            # Clear caches that depend on data
            generate_graph_data.clear()
            generate_plotly_figure.clear()
            st.session_state.update_graph = True
            st.success("Data direset ke nilai awal!")

with tabs[1]:
    st.subheader("Impor Data dari Excel")
    st.write("Unggah file Excel dengan nilai x dan y (harus memiliki kolom 'x_values' dan 'y_values')")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        imported_df = load_excel_data(uploaded_file) # Use cached function
        if imported_df is not None:
            st.write("Pratinjau Data:")
            st.dataframe(imported_df.head(), use_container_width=True)
            
            if st.button("Gunakan Data Ini", key="use_imported"):
                st.session_state.data = imported_df
                # Clear caches that depend on data
                generate_graph_data.clear()
                generate_plotly_figure.clear()
                st.session_state.update_graph = True
                st.success("Data yang diimpor berhasil diterapkan!")

# Display example Excel template for download
st.divider()
if st.button("Unduh Template Excel Contoh"):
    sample_df = pd.DataFrame(INITIAL_DATA)
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Klik untuk Mengunduh",
        data=buffer,
        file_name="thread_abrasion_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Create graph section
st.divider()
st.subheader("Grafik Abrasi Benang")

# Ambil data dari session state
x_values_current = st.session_state.data['x_values']
y_values_current = st.session_state.data['y_values']

# Panggil fungsi yang di-cache untuk menghasilkan data grafik (slope, intercept, dll)
# Sekarang hanya mengembalikan slope dan intercept dari regresi global
slope, intercept, y_line_intersection, y_at_x_50_curve = generate_graph_data(x_values_current, y_values_current)

# Panggil fungsi yang di-cache untuk membuat figure Plotly
fig = generate_plotly_figure(x_values_current, y_values_current, slope, intercept, y_line_intersection, y_at_x_50_curve)
    
# Display the graph
st.plotly_chart(fig, use_container_width=True)

# Display results in a cleaner format
st.markdown(f"""
<div class="dark-card">
    <h2 style="color: #FFFFFF; margin-bottom: 5px;">Hasil Analisis pada x=50</h2>
    <h1 style="color: #64CCC9; font-size: 48px; margin: 10px 0;">{y_line_intersection:.2f}</h1>
    <p style="color: #A0A0A0; font-size: 16px;">Nilai perpotongan garis regresi linear pada x=50</p>
    <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
        Berdasarkan garis kecocokan terbaik (regresi linear) dari <strong style="color: #64CCC9;">seluruh data</strong>.
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
        hide_index=False,
        use_container_width=True
    )

# Add important graph information
st.markdown("""
<div class="dark-card">
    <h3 style="color: #FFFFFF;">Analisis Abrasi Benang</h3>
    <p style="color: #F0F2F6;">Grafik menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y).</p>
    <ul style="color: #F0F2F6;">
        <li><strong style="color: #64CCC9;">Garis teal</strong> adalah kurva data abrasi benang.</li>
        <li><strong style="color: #FF6347;">Garis oranye/merah</strong> adalah <strong style="color: #FF6347;">garis regresi linear</strong> (garis kecocokan terbaik) yang mewakili tren dari seluruh data, dan diperpanjang ke x=50.</li>
        <li><strong style="color: #FF6347;">Lingkaran kosong</strong> menunjukkan perpotongan garis regresi dengan x=50.</li>
        <li><strong style="color: #DC3545;">Lingkaran padat</strong> menunjukkan perpotongan kurva data asli dengan x=50.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Add additional information in an expander
with st.expander("Informasi Tambahan & Tips Interaksi"):
    st.markdown("""
    <div style="padding: 10px;">
        <h4 style="color: #FFFFFF;">Poin Penting:</h4>
        <ul style="color: #F0F2F6;">
            <li>Garis oranye/merah kini mewakili <strong style="color: #FF6347;">garis regresi linear</strong> yang dihitung dari seluruh dataset, memberikan ekstrapolasi yang merepresentasikan tren keseluruhan data.</li>
            <li>Ini adalah garis yang secara statistik paling "mendekati" atau "melewati" tren dari semua titik data.</li>
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
