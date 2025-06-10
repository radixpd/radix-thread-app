import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate, stats
import io
from sklearn.linear_model import LinearRegression, RANSACRegressor # Import modul baru

# --- START OF ACCESS CODE IMPLEMENTATION ---
ACCESS_CODES = ["RADIX2025", "PULCRA2025", "ADMIN123", "GUEST456"]

def check_password():
    """Mengembalikan True jika pengguna memasukkan salah satu kode yang benar dari daftar, False jika tidak."""
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        st.subheader("Masukkan Kode Akses")
        password_input = st.text_input("Kode Akses", type="password", key="password_input")
        if st.button("Masuk", key="login_button"):
            if password_input in ACCESS_CODES:
                st.session_state.password_entered = True
                st.rerun() # PERBAIKAN: Menggunakan st.rerun()
            else:
                st.error("Kode akses salah. Silakan coba lagi.")
        return False
    return True

if not check_password():
    st.stop()
# --- END OF ACCESS CODE IMPLEMENTATION ---

st.set_page_config(
    page_title="Radix Thread Abrasion Graph",
    page_icon="📊",
    layout="wide"
)

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
        border-left: 4x solid #28A745 !important; /* Green for success */
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

    /* --- MOBILE FRIENDLY ADJUSTMENTS --- */
    @media (max-width: 768px) { /* Adjustments for tablets and smaller */
        .app-header h1 {
            font-size: 32px !important; /* Slightly smaller main title */
        }
        .pulcra-logo {
            font-size: 34px !important; /* Slightly smaller logo text */
        }
        .dark-card {
            padding: 20px !important; /* Slightly less padding on cards */
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding: 15px; /* Reduce tab content padding */
        }
    }

    @media (max-width: 600px) { /* Adjustments for typical phone portrait */
        .app-header {
            padding: 20px; /* Reduce header padding */
        }
        .app-header h1 {
            font-size: 28px !important; /* Smaller main title */
        }
        .pulcra-logo {
            font-size: 30px !important; /* Smaller logo text */
        }
        .dark-card {
            padding: 15px !important; /* Further reduce padding on cards */
        }
        .stButton>button {
            padding: 6px 12px !important; /* Slightly smaller buttons */
            font-size: 14px;
        }
        /* Ensure inputs and data editor are full width */
        [data-testid="stTextInput"], [data-testid="stDataEditor"] {
            width: 100% !important;
        }
    }

    @media (max-width: 400px) { /* Even smaller screens */
        .app-header h1 {
            font-size: 24px !important;
        }
        .pulcra-logo {
            font-size: 26px !important;
        }
        p, li, span {
            font-size: 14px; /* Smaller general text */
        }
    }
</style>
""", unsafe_allow_html=True)

# Header Aplikasi with Radix Logo
st.markdown("""
<div class="app-header">
    <div class="pulcra-logo">PULCRA CHEMICALS INDONESIA</div>
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
    df = pd.DataFrame(INITIAL_DATA)
    df.index = np.arange(1, len(df) + 1)
    return df

@st.cache_data
def load_excel_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'x_values' in df.columns and 'y_values' in df.columns:
                df = df[['x_values', 'y_values']]
                df.index = np.arange(1, len(df) + 1)
                return df
            else:
                st.error("File Excel harus berisi kolom 'x_values' dan 'y_values'.")
                return None
        except Exception as e:
            st.error(f"Error membaca file Excel: {e}")
            return None
    return None

@st.cache_data
def generate_graph_data(x_values, y_values, poly_degree, linear_model_type): # Tambahkan linear_model_type
    x_np = x_values.to_numpy().reshape(-1, 1) # Reshape untuk sklearn
    y_np = y_values.to_numpy()

    # Inisialisasi semua nilai perpotongan menjadi NaN
    ols_slope = np.nan
    ols_intercept = np.nan
    ransac_slope = np.nan
    ransac_intercept = np.nan
    y_at_x_50_ols = np.nan
    y_at_x_50_ransac = np.nan
    y_at_x_50_poly = np.nan
    y_at_x_50_curve = np.nan

    if y_values.empty:
        st.warning("Data kosong. Tidak dapat melakukan analisis.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Regresi Linear Biasa (OLS)
    if x_values.nunique() >= 2:
        ols_reg = LinearRegression()
        ols_reg.fit(x_np, y_np)
        ols_slope = ols_reg.coef_[0]
        ols_intercept = ols_reg.intercept_
        y_at_x_50_ols = ols_slope * 50 + ols_intercept
    else:
        st.warning("Tidak cukup variasi pada data X untuk melakukan regresi linear yang valid.")
        if len(x_values) >= 1:
            ols_intercept = y_values.mean() # Jika hanya 1 titik X, garis horizontal di rata-rata Y

    # Regresi Linear Robust (RANSAC)
    if x_values.nunique() >= 2 and len(x_np) > 1: # RANSAC membutuhkan setidaknya 2 titik
        try:
            ransac = RANSACRegressor(LinearRegression(), min_samples=2, random_state=42)
            ransac.fit(x_np, y_np)
            ransac_slope = ransac.estimator_.coef_[0]
            ransac_intercept = ransac.estimator_.intercept_
            y_at_x_50_ransac = ransac_slope * 50 + ransac_intercept
        except ValueError as e:
            st.warning(f"Tidak dapat melakukan Regresi RANSAC: {e}. Mungkin terlalu sedikit titik data atau titik data terlalu sedikit untuk inlier.")
            ransac_slope = np.nan
            ransac_intercept = np.nan
            y_at_x_50_ransac = np.nan
    

    # Hitung Regresi Polinomial
    if len(x_np) > poly_degree:
        poly_coeffs = np.polyfit(x_np.flatten(), y_np, poly_degree) # Gunakan flatten untuk np.polyfit
        poly_model = np.poly1d(poly_coeffs)
        y_at_x_50_poly = float(poly_model(50))
    else:
        y_at_x_50_poly = np.nan


    # Interpolasi Kurva Asli
    if len(x_np) >= 2:
        f_curve = interpolate.interp1d(x_np.flatten(), y_np, kind='linear', fill_value='extrapolate')
        y_at_x_50_curve = float(f_curve(50))
    else:
        y_at_x_50_curve = np.nan

    # Kembalikan semua nilai yang dihitung
    return ols_slope, ols_intercept, y_at_x_50_ols, \
           ransac_slope, ransac_intercept, y_at_x_50_ransac, \
           y_at_x_50_poly, y_at_x_50_curve

@st.cache_data
def generate_plotly_figure(x_values, y_values, ols_slope, ols_intercept, y_at_x_50_ols, \
                           ransac_slope, ransac_intercept, y_at_x_50_ransac, \
                           y_at_x_50_poly, y_at_x_50_curve, poly_degree, linear_model_type): # Tambahkan linear_model_type
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values,
        mode='lines+markers',
        name='Data Abrasi',
        line=dict(color='#64CCC9', width=3),
        marker=dict(size=8, color='#64CCC9')
    ))
    
    # Hitung rentang X untuk garis regresi
    x_line_min = x_values.min() if not x_values.empty else 0
    x_line_max = x_values.max() if not x_values.empty else 100 # Ekstrapolasi lebih jauh sedikit
    x_plot_range = np.linspace(x_line_min, max(x_line_max, 50), 100) # Range untuk plot regresi hingga setidaknya 50


    # Gambar Garis Regresi Linear (OLS) jika dipilih
    if linear_model_type == "Regresi Linear Biasa (OLS)":
        if not np.isnan(ols_slope) and not np.isnan(ols_intercept) and x_values.nunique() >= 2:
            y_ols_fit = ols_slope * x_plot_range + ols_intercept
            fig.add_trace(go.Scatter(
                x=x_plot_range,
                y=y_ols_fit,
                mode='lines',
                name='Garis Regresi Linear (OLS)',
                line=dict(color="#FF6347", width=3, dash="dot"),
            ))

            fig.add_trace(go.Scatter(
                x=[50],
                y=[y_at_x_50_ols],
                mode='markers',
                name=f'Perpotongan OLS pada x=50, y={y_at_x_50_ols:.2f}',
                marker=dict(size=14, color='#FF6347', symbol='circle-open', line=dict(width=3, color='#FF6347'))
            ))
            
            if not np.isnan(y_at_x_50_ols):
                fig.add_annotation(
                    x=50,
                    y=y_at_x_50_ols + 50,
                    text=f"OLS: {y_at_x_50_ols:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#FF6347',
                    font=dict(size=12, color='#FF6347', family="Arial, sans-serif"),
                )
    
    # Gambar Garis Regresi Linear Robust (RANSAC) jika dipilih
    elif linear_model_type == "Regresi Linear Robust (RANSAC)":
        if not np.isnan(ransac_slope) and not np.isnan(ransac_intercept) and x_values.nunique() >= 2:
            y_ransac_fit = ransac_slope * x_plot_range + ransac_intercept
            fig.add_trace(go.Scatter(
                x=x_plot_range,
                y=y_ransac_fit,
                mode='lines',
                name='Garis Regresi Linear Robust (RANSAC)',
                line=dict(color="#FFFF00", width=3, dash="dot"), # Kuning
            ))

            fig.add_trace(go.Scatter(
                x=[50],
                y=[y_at_x_50_ransac],
                mode='markers',
                name=f'Perpotongan RANSAC pada x=50, y={y_at_x_50_ransac:.2f}',
                marker=dict(size=14, color='#FFFF00', symbol='star-open', line=dict(width=3, color='#FFFF00')) # Bintang kuning
            ))
            
            if not np.isnan(y_at_x_50_ransac):
                fig.add_annotation(
                    x=50,
                    y=y_at_x_50_ransac + 50, # Sesuaikan posisi
                    text=f"RANSAC: {y_at_x_50_ransac:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='#FFFF00',
                    font=dict(size=12, color='#FFFF00', family="Arial, sans-serif"),
                )


    # Gambar Garis Regresi Polinomial (jika valid)
    if not np.isnan(y_at_x_50_poly) and len(x_values) > poly_degree:
        poly_coeffs = np.polyfit(x_values, y_values, poly_degree)
        poly_model = np.poly1d(poly_coeffs)
        
        x_poly_range = np.linspace(x_values.min(), max(x_values.max(), 50), 100) # Range hingga 50
        y_poly_fit = poly_model(x_poly_range)

        fig.add_trace(go.Scatter(
            x=x_poly_range,
            y=y_poly_fit,
            mode='lines',
            name=f'Regresi Polinomial Derajat {poly_degree}',
            line=dict(color='#8A2BE2', width=3, dash="dash"), # Warna ungu
        ))

        fig.add_trace(go.Scatter(
            x=[50],
            y=[y_at_x_50_poly],
            mode='markers',
            name=f'Perpotongan Regresi Polinomial pada x=50, y={y_at_x_50_poly:.2f}',
            marker=dict(size=14, color='#8A2BE2', symbol='square-open', line=dict(width=3, color='#8A2BE2')) # Kotak ungu
        ))
        
        if not np.isnan(y_at_x_50_poly):
            fig.add_annotation(
                x=50,
                y=y_at_x_50_poly - 50, # Posisikan di bawah titik
                text=f"Pol: {y_at_x_50_poly:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#8A2BE2',
                font=dict(size=12, color='#8A2BE2', family="Arial, sans-serif"),
            )


    max_y_val = max(y_values) if not y_values.empty else 1000
    fig.add_shape(
        type="line",
        x0=50,
        y0=0,
        x1=50,
        y1=max_y_val * 1.1,
        line=dict(color="#DC3545", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=50,
        y=max_y_val * 1.05,
        text="x=50",
        showarrow=False,
        font=dict(color="#DC3545", size=14, family="Arial, sans-serif", weight="bold")
    )
    
    if not np.isnan(y_at_x_50_curve):
        fig.add_trace(go.Scatter(
            x=[50],
            y=[y_at_x_50_curve],
            mode='markers',
            name=f'Perpotongan Kurva Asli pada x=50, y={y_at_x_50_curve:.2f}',
            marker=dict(size=14, color='#DC3545', symbol='circle', line=dict(width=2, color='white'))
        ))
        if not np.isnan(y_at_x_50_curve):
            fig.add_annotation(
                x=50,
                y=y_at_x_50_curve, # Posisikan di titik
                text=f"Asli: {y_at_x_50_curve:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#DC3545',
                font=dict(size=12, color='#DC3545', family="Arial, sans-serif"),
            )
    
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
            bgcolor="rgba(28,31,38,0.7)",
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
    st.write("Untuk mengganti angka di kolom **'y_value'**, **klik dua kali** pada sel yang ingin diubah, lalu masukkan angka baru. Setelah selesai, klik tombol **'Terapkan Perubahan'** di bawah tabel.")
    
    edited_data = pd.DataFrame({
        'x_value': st.session_state.data['x_values'],
        'y_value': st.session_state.data['y_values']
    })
    edited_data.index = np.arange(1, len(edited_data) + 1)

    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=False,
        use_container_width=True,
        key="data_editor",
        on_change=None
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes"):
            st.session_state.data['y_values'] = edited_df['y_value'].tolist()
            generate_graph_data.clear()
            generate_plotly_figure.clear()
            st.session_state.update_graph = True
            st.success("Data berhasil diperbarui!")
    
    with col2:
        if st.button("Reset ke Nilai Awal", key="reset_values"):
            st.session_state.data = get_initial_data_df()
            generate_graph_data.clear()
            generate_plotly_figure.clear()
            st.session_state.update_graph = True
            st.success("Data direset ke nilai awal!")

with tabs[1]:
    st.subheader("Impor Data dari Excel")
    st.write("Unggah file Excel dengan nilai x dan y (harus memiliki kolom 'x_values' dan 'y_values')")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        imported_df = load_excel_data(uploaded_file)
        if imported_df is not None:
            st.write("Pratinjau Data:")
            st.dataframe(imported_df, use_container_width=True, hide_index=False)
            
            if st.button("Gunakan Data Ini", key="use_imported"):
                st.session_state.data = imported_df
                generate_graph_data.clear()
                generate_plotly_figure.clear()
                st.session_state.update_graph = True
                st.success("Data yang diimpor berhasil diterapkan!")

st.divider()

# Kontrol untuk pilihan derajat polinomial dan jenis regresi linear
st.subheader("Opsi Kurva Analisis")
linear_model_type = st.radio(
    "Pilih Model Regresi Linear:",
    ("Regresi Linear Biasa (OLS)", "Regresi Linear Robust (RANSAC)"),
    help="Regresi Linear Biasa (OLS) cocok untuk data tanpa outlier ekstrem. Regresi Linear Robust (RANSAC) lebih baik jika ada outlier, karena fokus pada 'mayoritas' titik data."
)

poly_degree = st.slider(
    "Pilih Derajat Regresi Polinomial (untuk kurva kecocokan yang lebih dekat):",
    min_value=1,
    max_value=min(5, len(st.session_state.data['x_values']) - 1), # Max degree is data points - 1
    value=1 if len(st.session_state.data['x_values']) < 2 else 2, # Default to 1 (linear) if few points, else 2
    step=1,
    help="Derajat 1 adalah regresi linear. Derajat yang lebih tinggi akan menghasilkan kurva yang lebih dekat dengan titik data, tetapi bisa 'overfit'."
)
if poly_degree >= len(st.session_state.data['x_values']):
    st.warning(f"Derajat polinomial ({poly_degree}) terlalu tinggi untuk jumlah titik data ({len(st.session_state.data['x_values'])}). Kurva polinomial mungkin tidak akurat atau tidak dapat dihitung.")


st.subheader("Grafik Abrasi Benang")

x_values_current = st.session_state.data['x_values']
y_values_current = st.session_state.data['y_values']

ols_slope, ols_intercept, y_ols_intersection, \
ransac_slope, ransac_intercept, y_ransac_intersection, \
y_poly_intersection, y_at_x_50_curve = generate_graph_data(x_values_current, y_values_current, poly_degree, linear_model_type) # Teruskan parameter baru

fig = generate_plotly_figure(x_values_current, y_values_current, ols_slope, ols_intercept, y_ols_intersection, \
                             ransac_slope, ransac_intercept, y_ransac_intersection, \
                             y_poly_intersection, y_at_x_50_curve, poly_degree, linear_model_type) # Teruskan parameter baru
    
st.plotly_chart(fig, use_container_width=True)

# Menampilkan hasil analisis untuk setiap jenis kurva
st.markdown(f"""
<div class="dark-card">
    <h2 style="color: #FFFFFF; margin-bottom: 5px;">Hasil Analisis pada x=50</h2>
    """)

if linear_model_type == "Regresi Linear Biasa (OLS)":
    if not np.isnan(y_ols_intersection):
        st.markdown(f"""
        <p style="color: #F0F2F6; font-size: 16px; margin-bottom: 5px;">
            **Perpotongan Garis Regresi Linear (OLS):** <span style="font-weight: bold; color: #FF6347; font-size: 20px;">{y_ols_intersection:.2f}</span>
            <br><span style="font-size: 14px; color: #A0A0A0;">(Berdasarkan garis lurus kecocokan terbaik dari seluruh data)</span>
        </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="color: #F0F2F6; font-size: 16px; margin-bottom: 5px;">
            **Perpotongan Garis Regresi Linear (OLS):** <span style="font-weight: bold; color: #FF6347; font-size: 20px;">Tidak dapat dihitung</span>
            <br><span style="font-size: 14px; color: #A0A0A0;">(Perlu setidaknya 2 titik X yang berbeda)</span>
        </p>
        """, unsafe_allow_html=True)
elif linear_model_type == "Regresi Linear Robust (RANSAC)":
    if not np.isnan(y_ransac_intersection):
        st.markdown(f"""
        <p style="color: #F0F2F6; font-size: 16px; margin-bottom: 5px;">
            **Perpotongan Garis Regresi Linear Robust (RANSAC):** <span style="font-weight: bold; color: #FFFF00; font-size: 20px;">{y_ransac_intersection:.2f}</span>
            <br><span style="font-size: 14px; color: #A0A0A0;">(Berdasarkan garis lurus yang kurang terpengaruh outlier)</span>
        </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="color: #F0F2F6; font-size: 16px; margin-bottom: 5px;">
            **Perpotongan Garis Regresi Linear Robust (RANSAC):** <span style="font-weight: bold; color: #FFFF00; font-size: 20px;">Tidak dapat dihitung</span>
            <br><span style="font-size: 14px; color: #A0A0A0;">(Perlu setidaknya 2 titik X yang berbeda dan cukup inlier)</span>
        </p>
        """, unsafe_allow_html=True)


if not np.isnan(y_poly_intersection):
    st.markdown(f"""
    <p style="color: #F0F2F6; font-size: 16px; margin-bottom: 5px;">
        **Perpotongan Regresi Polinomial Derajat {poly_degree}:** <span style="font-weight: bold; color: #8A2BE2; font-size: 20px;">{y_poly_intersection:.2f}</span>
        <br><span style="font-size: 14px; color: #A0A0A0;">(Berdasarkan kurva polinomial derajat {poly_degree} yang menyesuaikan data)</span>
    </p>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <p style="color: #F0F2F6; font-size: 16px; margin-bottom: 5px;">
        **Perpotongan Regresi Polinomial Derajat {poly_degree}:** <span style="font-weight: bold; color: #8A2BE2; font-size: 20px;">Tidak dapat dihitung</span>
        <br><span style="font-size: 14px; color: #A0A0A0;">(Perlu lebih dari {poly_degree} titik data unik untuk derajat ini)</span>
    </p>
    """, unsafe_allow_html=True)

if not np.isnan(y_at_x_50_curve):
    st.markdown(f"""
    <p style="color: #F0F2F6; font-size: 16px;">
        **Perpotongan Kurva Data Asli (Interpolasi Linear):** <span style="font-weight: bold; color: #DC3545; font-size: 20px;">{y_at_x_50_curve:.2f}</span>
        <br><span style="font-size: 14px; color: #A0A0A0;">(Berdasarkan interpolasi linear antar titik data yang sudah ada)</span>
    </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <p style="color: #F0F2F6; font-size: 16px;">
        **Perpotongan Kurva Data Asli:** <span style="font-weight: bold; color: #DC3545; font-size: 20px;">Tidak dapat dihitung</span>
        <br><span style="font-size: 14px; color: #A0A0A0;">(Perlu setidaknya 2 titik data untuk interpolasi)</span>
    </p>
    </div>
    """, unsafe_allow_html=True)


with st.expander("Lihat Tabel Data Lengkap"):
    display_df = pd.DataFrame({
        'x_values (Nilai yang sudah tetap)': st.session_state.data['x_values'],
        'y_values (N atau nilai benang putus)': st.session_state.data['y_values']
    })
    display_df.index = np.arange(1, len(display_df) + 1)
    st.dataframe(
        display_df,
        hide_index=False,
        use_container_width=True
    )

st.markdown("""
<div class="dark-card">
    <h3 style="color: #FFFFFF;">Analisis Abrasi Benang</h3>
    <p style="color: #F0F2F6;">Grafik menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y).</p>
    <ul style="color: #F0F2F6;">
        <li><strong style="color: #64CCC9;">Garis teal</strong> adalah kurva data abrasi benang yang menghubungkan setiap titik data Anda.</li>
        <li><strong style="color: #FF6347;">Garis oranye/merah putus-putus</strong> adalah <strong style="color: #FF6347;">garis regresi linear biasa (OLS)</strong>. Ini adalah garis lurus yang mewakili tren umum data, sensitif terhadap outlier.</li>
        <li><strong style="color: #FFFF00;">Garis kuning putus-putus</strong> adalah <strong style="color: #FFFF00;">garis regresi linear robust (RANSAC)</strong>. Ini juga garis lurus, tetapi dirancang untuk lebih baik melewati "mayoritas" titik data dengan mengabaikan outlier.</li>
        <li><strong style="color: #8A2BE2;">Garis ungu putus-putus</strong> adalah <strong style="color: #8A2BE2;">garis regresi polinomial</strong>. Ini adalah kurva yang dapat menyesuaikan diri dengan bentuk data yang lebih kompleks.</li>
        <li><strong style="color: #FF6347;">Lingkaran kosong (oranye)</strong> menunjukkan perpotongan garis regresi linear biasa (OLS) dengan x=50.</li>
        <li><strong style="color: #FFFF00;">Bintang kosong (kuning)</strong> menunjukkan perpotongan garis regresi linear robust (RANSAC) dengan x=50.</li>
        <li><strong style="color: #8A2BE2;">Kotak kosong (ungu)</strong> menunjukkan perpotongan garis regresi polinomial dengan x=50.</li>
        <li><strong style="color: #DC3545;">Lingkaran padat (merah)</strong> menunjukkan perpotongan kurva data asli (garis teal) dengan x=50.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

with st.expander("Informasi Tambahan & Tips Interaksi"):
    st.markdown("""
    <div style="padding: 10px;">
        <h4 style="color: #FFFFFF;">Poin Penting:</h4>
        <ul style="color: #F0F2F6;">
            <li>Garis regresi linear biasa (OLS, oranye) adalah garis lurus yang mewakili tren statistik terbaik.</li>
            <li><strong style="color: #FFFF00;">Garis regresi linear robust (RANSAC, kuning)</strong> adalah garis lurus yang berusaha untuk "memotong" bagian utama dari data Anda, mengabaikan titik-titik yang dianggap outlier. Coba ganti pilihan di "Pilih Model Regresi Linear" untuk melihat perbedaannya.</li>
            <li>Garis regresi polinomial (ungu) adalah kurva yang dapat mengikuti pola data Anda lebih dekat. Anda bisa memilih derajatnya.</li>
            <li>Kurva data asli (teal) secara visual menghubungkan titik-titik data, memberikan representasi langsung dari data yang Anda masukkan.</li>
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

st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 15px; font-size: 14px; font-family: 'Helvetica Neue', sans-serif; color: #A0A0A0; border-top: 1px solid #2C303A; background-color: #1C1F26; border-radius: 0 0 8px 8px;">
    <p>Dikembangkan oleh <span style="font-weight: bold; color: #64CCC9;">RADIX</span> &copy; 2025</p>
    <p>Solusi Analisis Abrasi Benang Profesional</p>
</div>
""", unsafe_allow_html=True)
