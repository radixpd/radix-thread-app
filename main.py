import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor, LinearRegression
import io

# --- Konfigurasi Halaman (Paling Awal) ---
st.set_page_config(
    page_title="Thread Abrasion by Radix",
    page_icon="🧵",
    layout="wide"
)

# --- CSS Kustom untuk Tampilan Dark Mode Minimalis & Elegan (Revisi Tambahan untuk Responsif) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap');

    /* Target elemen HTML dan body untuk memastikan background hitam total */
    html, body {
        background-color: #0A0A0A !important;
        color: #FFFFFF !important; /* Teks lebih terang */
    }

    /* Streamlit's main wrapper */
    .stApp {
        background-color: #0A0A0A !important;
        max-width: 1300px;
        margin: 0 auto;
        padding-top: 30px;
        padding-bottom: 50px;
        padding-left: 15px;
        padding-right: 15px;
    }
    
    /* Main content area within .stApp */
    .main {
        background-color: #0A0A0A;
        color: #FFFFFF;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Kontainer utama untuk sidebar jika ada */
    .stSidebar {
        background-color: #0A0A0A !important;
        color: #FFFFFF;
    }
    
    .block-container {
        background-color: #0A0A0A !important;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Typography - Improved contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.8px;
        word-break: break-word;
    }
    h1 {
        font-weight: 700;
        font-size: 44px;
        padding-bottom: 15px;
        border-bottom: 3px solid #8B4513;
        text-align: center;
        text-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    h2 {
        font-weight: 600;
        font-size: 32px;
        color: #FFD700 !important; /* Gold color for better visibility */
        margin-bottom: 20px;
        border-bottom: 1px solid #444;
        padding-bottom: 8px;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 0.5px;
    }
    h3 {
        font-weight: 600;
        font-size: 24px;
        color: #FFFFFF !important;
        font-family: 'Montserrat', sans-serif;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    p, li, span, div {
        color: #FFFFFF !important;
        font-family: 'Montserrat', sans-serif;
        line-height: 1.8;
        font-size: 17px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #8B4513;
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 12px 25px;
        font-size: 17px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .stButton>button:hover {
        background-color: #A0522D;
        box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
        transform: translateY(-3px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Montserrat', sans-serif;
        color: #FFFFFF !important;
        font-weight: 600;
        padding: 12px 20px;
        font-size: 17px;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 12px;
        background-color: #1A1A1A;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        border: 1px solid #444;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 30px;
        border-radius: 12px;
        background-color: #1A1A1A;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FFD700;
        border-radius: 6px;
        height: 4px;
    }

    /* Radio Buttons */
    .stRadio > label {
        color: #FFFFFF !important;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stRadio > div {
        background-color: #1A1A1A;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
    }
    .stRadio [data-baseweb="radio"] {
        background-color: #282828;
        border-radius: 10px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, border 0.3s ease;
        flex-grow: 1;
        text-align: center;
        min-width: 150px;
    }
    .stRadio [data-baseweb="radio"]:hover {
        background-color: #3A3A3A;
        border: 1px solid #FFD700;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: #FFD700 !important;
        color: #000000 !important;
        border: 1px solid #FFD700;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
    }
    .stRadio [data-baseweb="radio"] span:last-child {
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 17px;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: #000000 !important;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1A1A1A;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        margin-bottom: 30px;
        border: 1px solid #444;
    }
    .dark-card.result-card {
        height: auto !important;
        overflow: visible !important;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 25px;
    }
    .dark-card.result-card h3 {
        margin-bottom: 10px;
        text-align: center;
    }
    .dark-card.result-card p {
        text-align: center;
    }
    .dark-card.result-card p:last-child {
        margin-top: 10px;
        font-size: 13px;
    }

    /* Radix Header */
    .app-header {
        background: linear-gradient(145deg, #1A1A1A, #0A0A0A);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(8px);
        border: 1px solid #444;
    }
    .pulcra-logo {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 56px;
        color: #FFD700;
        margin-bottom: 10px;
        letter-spacing: 5px;
        text-shadow: 0 5px 20px rgba(255, 215, 0, 0.5);
        text-transform: uppercase;
    }
    .app-header h1 {
        font-size: 38px;
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0;
        text-shadow: none;
        color: #FFFFFF;
    }
    .app-header p {
        font-size: 18px;
        color: #CCCCCC;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }

    /* Footer */
    .radix-footer {
        text-align: center;
        margin-top: 60px;
        padding: 25px;
        font-size: 15px;
        font-family: 'Montserrat', sans-serif;
        color: #CCCCCC;
        border-top: 1px solid #444;
        background-color: #1A1A1A;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.3);
    }

    /* Other elements */
    hr {
        border-color: #444 !important;
        margin: 40px 0 !important;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    
    /* Streamlit specific adjustments */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0px !important;
        position: fixed !important;
    }
    .stApp:hover [data-testid="stToolbar"] {
        visibility: visible !important;
        height: auto !important;
    }

    /* For data editor and file uploader */
    [data-testid="stDataEditor"] {
        border-radius: 10px;
        overflow: auto;
        border: 1px solid #444;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #FFD700;
        border-radius: 12px;
        padding: 25px;
        background-color: #1A1A1A;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploaderDropzone"] p {
        color: #CCCCCC;
        font-size: 17px;
    }

    /* For data preview tables */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.3);
        background-color: #1A1A1A;
        max-height: 350px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #444;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
        min-width: 600px;
    }
    .stDataFrame th {
        background-color: #282828 !important;
        color: #FFD700 !important;
        font-weight: 700;
        position: sticky;
        top: 0;
        z-index: 1;
        font-size: 16px;
    }
    .stDataFrame td {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border-bottom: 1px solid #444 !important;
        padding: 10px 15px;
    }
    /* Scrollbar for dataframes */
    .stDataFrame::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    .stDataFrame::-webkit-scrollbar-track {
        background: #1A1A1A;
    }
    .stDataFrame::-webkit-scrollbar-thumb {
        background: #FFD700;
        border-radius: 10px;
    }
    .stDataFrame::-webkit-scrollbar-thumb:hover {
        background: #C49F3D;
    }

    /* Plotly specifics */
    .js-plotly-plot .plotly .modebar {
        background-color: #1A1A1A !important;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .js-plotly-plot .plotly .modebar-btn {
        color: #FFD700 !important;
    }
    .js-plotly-plot .plotly .modebar-btn:hover {
        background-color: #282828 !important;
    }

    /* Access Code styling */
    .stTextInput>div>div>input {
        background-color: #1A1A1A;
        border: 1px solid #444;
        border-radius: 8px;
        color: #FFFFFF;
        padding: 10px 15px;
        font-size: 18px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput>label {
        font-size: 18px;
        color: #FFFFFF !important;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .st-emotion-cache-16txt4s {
        background-color: #4A0000;
        color: #FFCCCC;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #8B0000;
    }
    .st-emotion-cache-zt5ig8 {
        background-color: #004A00;
        color: #CCFFCC;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #008B00;
    }

    /* --- MEDIA QUERIES FOR MOBILE RESPONSIVENESS --- */
    @media (max-width: 768px) {
        .stApp {
            padding-left: 10px;
            padding-right: 10px;
            padding-top: 20px;
            padding-bottom: 30px;
        }
        h1 {
            font-size: 32px;
            padding-bottom: 10px;
        }
        h2 {
            font-size: 24px;
            margin-bottom: 15px;
        }
        h3 {
            font-size: 20px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        p, li, span, div {
            font-size: 15px;
            line-height: 1.6;
        }
        .stButton>button {
            padding: 10px 20px;
            font-size: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 15px;
            font-size: 15px;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding: 20px;
        }
        .stRadio > label {
            font-size: 16px;
            margin-bottom: 10px;
        }
        .stRadio > div {
            flex-direction: column;
            align-items: stretch;
            padding: 15px;
        }
        .stRadio [data-baseweb="radio"] {
            min-width: unset;
            padding: 8px 15px;
            font-size: 15px;
            width: 100%;
        }
        .app-header {
            padding: 30px;
            margin-bottom: 30px;
        }
        .pulcra-logo {
            font-size: 40px;
            letter-spacing: 3px;
        }
        .app-header h1 {
            font-size: 28px;
        }
        .app-header p {
            font-size: 16px;
        }
        .dark-card {
            padding: 20px;
            margin-bottom: 20px;
        }
        .stDataFrame th, .stDataFrame td {
            font-size: 14px;
            padding: 8px 10px;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 8px 12px;
        }
        .stTextInput>label {
            font-size: 16px;
        }
        .radix-footer {
            margin-top: 40px;
            padding: 15px;
            font-size: 13px;
        }
    }

    @media (max-width: 480px) {
        h1 {
            font-size: 28px;
        }
        h2 {
            font-size: 20px;
        }
        p, li, span, div {
            font-size: 14px;
        }
        .stButton>button {
            padding: 8px 15px;
            font-size: 14px;
        }
        .pulcra-logo {
            font-size: 32px;
        }
        .app-header h1 {
            font-size: 24px;
        }
        .app-header p {
            font-size: 14px;
        }
        .dark-card {
            padding: 15px;
        }
        .stDataFrame th, .stDataFrame td {
            font-size: 13px;
        }
    }

</style>
""", unsafe_allow_html=True)

# --- Implementasi Kode Akses ---
ACCESS_CODE = "RADIX2025"
TARGET_X_VALUE = 50 # Definisi konstanta untuk nilai X target

def check_password():
    if not st.session_state.get('password_entered', False):
        st.markdown("<h2 style='text-align: center;'>Akses Aplikasi</h2>", unsafe_allow_html=True)
        password_input = st.text_input("Masukkan Kode Akses Anda", type="password", key="password_input", help="Hubungi administrator untuk kode akses.")
        
        col_pw1, col_pw2, col_pw3 = st.columns([1,1,1])
        with col_pw2:
            if st.button("Masuk", key="login_button", use_container_width=True):
                if password_input == ACCESS_CODE:
                    st.session_state.password_entered = True
                    st.success("Akses berhasil! Memuat aplikasi...")
                    st.rerun()
                else:
                    st.error("Kode akses salah. Silakan coba lagi.")
        st.markdown("<br><br>", unsafe_allow_html=True)
        return False
    return True

# --- Data Awal ---
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# --- Inisialisasi Session State (Dipusatkan) ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)
if 'calculated_results' not in st.session_state:
    st.session_state.calculated_results = {}
if 'password_entered' not in st.session_state:
    st.session_state.password_entered = False
if 'data_needs_recalc' not in st.session_state:
    st.session_state.data_needs_recalc = True

# Panggil fungsi pengecekan password di awal aplikasi
if not check_password():
    st.stop()

# --- Header Aplikasi ---
st.markdown("""
<div class="app-header">
    <div class="pulcra-logo">PULCRA</div>
    <h1 style="margin-top: 0;">Analisis Abrasi Benang</h1>
    <p>Alat profesional untuk visualisasi data dan perhitungan nilai perpotongan</p>
</div>
""", unsafe_allow_html=True)

# --- Fungsi untuk Menghitung Garis dan Titik (dengan Cache) ---
@st.cache_data(show_spinner="Menghitung analisis data...")
def calculate_lines_and_points(x_values_series, y_values_series):
    results = {
        'y_at_x_50_original_curve': np.nan,
        'specific_x1_pt10_20': np.nan, 'specific_y1_pt10_20': np.nan,
        'specific_x2_pt10_20': np.nan, 'specific_y2_pt10_20': np.nan,
        'y_at_x_50_pt10_20_line': np.nan,
        'pt10_20_line_x_range': np.array([]), 'pt10_20_line_y': np.array([]),
        'y_at_x_50_ransac_line': np.nan,
        'ransac_line_x': np.array([]), 'ransac_line_y': np.array([])
    }

    # Konversi ke numpy array untuk interpolasi
    x_np = x_values_series.values
    y_np = y_values_series.values

    if len(x_np) < 2 or len(y_np) < 2:
        st.warning("Data tidak cukup untuk analisis. Masukkan minimal 2 pasangan X dan Y.")
        return results

    # Original curve interpolation
    try:
        # Pastikan x_np monoton meningkat untuk interp1d
        if not np.all(np.diff(x_np) > 0):
            st.error("Nilai 'x_values' harus monoton meningkat untuk interpolasi kurva. Harap perbaiki data Anda.")
            return results
        f = interpolate.interp1d(x_np, y_np, kind='linear', fill_value='extrapolate')
        results['y_at_x_50_original_curve'] = float(f(TARGET_X_VALUE))
    except ValueError as e:
        st.warning(f"Tidak dapat melakukan interpolasi kurva asli: {e}. Periksa data X Anda.")
        pass # Biarkan NaN jika gagal

    # Garis Antara Titik 10 & 20
    if len(x_np) >= 20:
        results['specific_x1_pt10_20'] = x_np[9] # Indeks 9 adalah titik ke-10
        results['specific_y1_pt10_20'] = y_np[9]
        results['specific_x2_pt10_20'] = x_np[19] # Indeks 19 adalah titik ke-20
        results['specific_y2_pt10_20'] = y_np[19]
    elif len(x_np) >= 2:
        # Fallback: gunakan titik pertama dan terakhir jika kurang dari 20
        results['specific_x1_pt10_20'] = x_np[0]
        results['specific_y1_pt10_20'] = y_np[0]
        results['specific_x2_pt10_20'] = x_np[-1]
        results['specific_y2_pt10_20'] = y_np[-1]
    
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']):
        if results['specific_x1_pt10_20'] != results['specific_x2_pt10_20']:
            slope_pt10_20 = (results['specific_y2_pt10_20'] - results['specific_y1_pt10_20']) / (results['specific_x2_pt10_20'] - results['specific_x1_pt10_20'])
            intercept_pt10_20 = results['specific_y1_pt10_20'] - slope_pt10_20 * results['specific_x1_pt10_20']
            results['y_at_x_50_pt10_20_line'] = slope_pt10_20 * TARGET_X_VALUE + intercept_pt10_20

            x_min_plot = x_np.min() if x_np.size > 0 else 0
            x_max_plot = x_np.max() if x_np.size > 0 else 100
            results['pt10_20_line_x_range'] = np.linspace(min(x_min_plot, TARGET_X_VALUE), max(x_max_plot, TARGET_X_VALUE), 100)
            results['pt10_20_line_y'] = slope_pt10_20 * results['pt10_20_line_x_range'] + intercept_pt10_20
        else:
            # Jika x1 dan x2 sama, garis vertikal atau titik tunggal, tidak dapat dihitung kemiringan
            results['y_at_x_50_pt10_20_line'] = np.nan
            results['pt10_20_line_x_range'] = np.array([])
            results['pt10_20_line_y'] = np.array([])

    # Regresi Linear Robust (RANSAC)
    if len(x_np) >= 2:
        try:
            X_reshaped = x_np.reshape(-1, 1)
            
            # Hitung residual_threshold secara dinamis, pastikan tidak nol
            residual_threshold_val = np.std(y_np) * 0.5 if len(y_np) > 1 and np.std(y_np) > 0 else 1.0
            
            ransac = RANSACRegressor(LinearRegression(),
                                     min_samples=2,
                                     residual_threshold=residual_threshold_val,
                                     random_state=42,
                                     max_trials=1000)
            ransac.fit(X_reshaped, y_np)
            results['y_at_x_50_ransac_line'] = ransac.predict(np.array([[TARGET_X_VALUE]]))[0]

            x_min_plot = x_np.min() if x_np.size > 0 else 0
            x_max_plot = x_np.max() if x_np.size > 0 else 100
            
            results['ransac_line_x'] = np.linspace(min(x_min_plot, TARGET_X_VALUE), max(x_max_plot, TARGET_X_VALUE), 100)
            results['ransac_line_y'] = ransac.predict(results['ransac_line_x'].reshape(-1, 1))
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menghitung regresi RANSAC: {e}")
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])
            results['y_at_x_50_ransac_line'] = np.nan

    return results

# Function to generate the Plotly graph with results in the top-left corner
def create_abrasion_plot(x_values, y_values, results, analysis_choice):
    fig = go.Figure()

    # Add Abrasion Data Curve (always)
    if not x_values.empty and not y_values.empty:
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=y_values,
            mode='lines+markers',
            name='Data Abrasi',
            line=dict(color='#FFD700', width=3),  # Changed to gold for better visibility
            marker=dict(size=8, color='#FFA500')  # Orange markers
        ))

        # Add Vertical Line at x=50 (always)
        plot_y_min = y_values.min() if not y_values.empty else 0
        plot_y_max = y_values.max() if not y_values.empty else 1000
        y_range_span = plot_y_max - plot_y_min
        y0_line = plot_y_min - y_range_span * 0.1 if y_range_span > 0 else 0
        y1_line = plot_y_max + y_range_span * 0.1 if y_range_span > 0 else 1000

        fig.add_shape(
            type="line",
            x0=TARGET_X_VALUE, y0=y0_line,
            x1=TARGET_X_VALUE, y1=y1_line,
            line=dict(color="#FF6347", width=2, dash="dash"),  # Tomato color
            layer="below"
        )
        fig.add_annotation(
            x=TARGET_X_VALUE, y=y1_line * 0.95,
            text=f"x={TARGET_X_VALUE}", showarrow=False,
            font=dict(color="#FF6347", size=14, family="Montserrat, sans-serif", weight="bold"),
            bgcolor="rgba(26,26,26,0.7)", bordercolor="#FF6347", borderwidth=1, borderpad=4
        )

        # Add specific lines based on exact choice
        if analysis_choice == "Garis Titik 10 & 20":
            if results.get('pt10_20_line_x_range', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['pt10_20_line_x_range'],
                    y=results['pt10_20_line_y'],
                    mode='lines',
                    name='Garis Titik 10 & 20',
                    line=dict(color='#00BFFF', width=2, dash='dot')  # Deep sky blue
                ))
                # Add points for 10th and 20th data point if they exist
                if not np.isnan(results['specific_x1_pt10_20']):
                    fig.add_trace(go.Scatter(
                        x=[results['specific_x1_pt10_20']],
                        y=[results['specific_y1_pt10_20']],
                        mode='markers',
                        name='Titik ke-10',
                        marker=dict(size=10, color='#00BFFF', symbol='circle')
                    ))
                if not np.isnan(results['specific_x2_pt10_20']):
                    fig.add_trace(go.Scatter(
                        x=[results['specific_x2_pt10_20']],
                        y=[results['specific_y2_pt10_20']],
                        mode='markers',
                        name='Titik ke-20',
                        marker=dict(size=10, color='#00BFFF', symbol='circle')
                    ))
            if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE], y=[results['y_at_x_50_pt10_20_line']],
                    mode='markers',
                    name='Potongan Garis 10-20 di x=50',
                    marker=dict(size=12, color='#00BFFF', symbol='star'),
                    hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"
                ))
        
        elif analysis_choice == "Garis yang melewati banyak titik":
            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['ransac_line_x'],
                    y=results['ransac_line_y'],
                    mode='lines',
                    name='Regresi RANSAC',
                    line=dict(color='#32CD32', width=2, dash='dash')  # Lime green
                ))
            if not np.isnan(results.get('y_at_x_50_ransac_line')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE], y=[results['y_at_x_50_ransac_line']],
                    mode='markers',
                    name='Potongan RANSAC di x=50',
                    marker=dict(size=12, color='#32CD32', symbol='star'),
                    hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"
                ))

        elif analysis_choice == "Kurva Data Asli":
            if not np.isnan(results.get('y_at_x_50_original_curve')):
                 fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE], y=[results['y_at_x_50_original_curve']],
                    mode='markers',
                    name='Potongan Kurva Asli di x=50',
                    marker=dict(size=12, color='#FFD700', symbol='star'),
                    hovertemplate=f"<b>Potongan (Kurva Asli)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"
                ))
        
        elif analysis_choice == "Tampilkan Semua":
            # Show all lines if "Tampilkan Semua" is selected
            if results.get('pt10_20_line_x_range', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['pt10_20_line_x_range'],
                    y=results['pt10_20_line_y'],
                    mode='lines',
                    name='Garis Titik 10 & 20',
                    line=dict(color='#00BFFF', width=2, dash='dot')
                ))
                if not np.isnan(results['specific_x1_pt10_20']):
                    fig.add_trace(go.Scatter(x=[results['specific_x1_pt10_20']], y=[results['specific_y1_pt10_20']], mode='markers', name='Titik ke-10', marker=dict(size=10, color='#00BFFF', symbol='circle')))
                if not np.isnan(results['specific_x2_pt10_20']):
                    fig.add_trace(go.Scatter(x=[results['specific_x2_pt10_20']], y=[results['specific_y2_pt10_20']], mode='markers', name='Titik ke-20', marker=dict(size=10, color='#00BFFF', symbol='circle')))
                if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
                    fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_pt10_20_line']], mode='markers', name='Potongan Garis 10-20 di x=50', marker=dict(size=12, color='#00BFFF', symbol='star'), hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))

            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['ransac_line_x'],
                    y=results['ransac_line_y'],
                    mode='lines',
                    name='Regresi RANSAC',
                    line=dict(color='#32CD32', width=2, dash='dash')
                ))
            if not np.isnan(results.get('y_at_x_50_ransac_line')):
                fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_ransac_line']], mode='markers', name='Potongan RANSAC di x=50', marker=dict(size=12, color='#32CD32', symbol='star'), hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))
            
            if not np.isnan(results.get('y_at_x_50_original_curve')):
                 fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_original_curve']], mode='markers', name='Potongan Kurva Asli di x=50', marker=dict(size=12, color='#FFD700', symbol='star'), hovertemplate=f"<b>Potongan (Kurva Asli)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))

    # Update layout for dark mode with results in top-left corner
    fig.update_layout(
        title={
            'text': 'Grafik Abrasi Benang',
            'yref': 'paper', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(color='#FFFFFF', size=24, family='Playfair Display, serif')
        },
        xaxis_title='Nilai X',
        yaxis_title='Nilai Benang Putus (N)',
        plot_bgcolor='#1A1A1A',
        paper_bgcolor='#1A1A1A',
        font=dict(color='#FFFFFF', family='Montserrat, sans-serif'),
        xaxis=dict(
            showgrid=True, gridcolor='#444', zeroline=False,
            title_font=dict(size=18), tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#444', zeroline=False,
            title_font=dict(size=18), tickfont=dict(size=14)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(26,26,26,0.7)", bordercolor="#444", borderwidth=1,
            font=dict(size=14)
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, b=40, t=100)
    )
    
    # Add results annotations in top-left corner
    if analysis_choice == "Kurva Data Asli" and not np.isnan(results.get('y_at_x_50_original_curve')):
        fig.add_annotation(
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            text=f"<b>Kurva Asli:</b> {results['y_at_x_50_original_curve']:.2f} N",
            showarrow=False,
            font=dict(size=14, color="#FFD700"),
            bgcolor="rgba(26,26,26,0.7)",
            bordercolor="#FFD700",
            borderwidth=1,
            borderpad=4
        )
    elif analysis_choice == "Garis Titik 10 & 20" and not np.isnan(results.get('y_at_x_50_pt10_20_line')):
        fig.add_annotation(
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            text=f"<b>Garis 10-20:</b> {results['y_at_x_50_pt10_20_line']:.2f} N",
            showarrow=False,
            font=dict(size=14, color="#00BFFF"),
            bgcolor="rgba(26,26,26,0.7)",
            bordercolor="#00BFFF",
            borderwidth=1,
            borderpad=4
        )
    elif analysis_choice == "Garis yang melewati banyak titik" and not np.isnan(results.get('y_at_x_50_ransac_line')):
        fig.add_annotation(
            x=0.05, y=0.95,
            xref="paper", yref="paper",
            text=f"<b>RANSAC:</b> {results['y_at_x_50_ransac_line']:.2f} N",
            showarrow=False,
            font=dict(size=14, color="#32CD32"),
            bgcolor="rgba(26,26,26,0.7)",
            bordercolor="#32CD32",
            borderwidth=1,
            borderpad=4
        )
    elif analysis_choice == "Tampilkan Semua":
        y_pos = 0.95
        if not np.isnan(results.get('y_at_x_50_original_curve')):
            fig.add_annotation(
                x=0.05, y=y_pos,
                xref="paper", yref="paper",
                text=f"<b>Kurva Asli:</b> {results['y_at_x_50_original_curve']:.2f} N",
                showarrow=False,
                font=dict(size=14, color="#FFD700"),
                bgcolor="rgba(26,26,26,0.7)",
                bordercolor="#FFD700",
                borderwidth=1,
                borderpad=4
            )
            y_pos -= 0.08
        
        if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
            fig.add_annotation(
                x=0.05, y=y_pos,
                xref="paper", yref="paper",
                text=f"<b>Garis 10-20:</b> {results['y_at_x_50_pt10_20_line']:.2f} N",
                showarrow=False,
                font=dict(size=14, color="#00BFFF"),
                bgcolor="rgba(26,26,26,0.7)",
                bordercolor="#00BFFF",
                borderwidth=1,
                borderpad=4
            )
            y_pos -= 0.08
        
        if not np.isnan(results.get('y_at_x_50_ransac_line')):
            fig.add_annotation(
                x=0.05, y=y_pos,
                xref="paper", yref="paper",
                text=f"<b>RANSAC:</b> {results['y_at_x_50_ransac_line']:.2f} N",
                showarrow=False,
                font=dict(size=14, color="#32CD32"),
                bgcolor="rgba(26,26,26,0.7)",
                bordercolor="#32CD32",
                borderwidth=1,
                borderpad=4
            )
    
    return fig

# --- Bagian Input Data ---
st.subheader("Input Data")
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.write("Masukkan data abrasi ke tabel Nilai Benang Putus. **Nilai X tetap** dan tidak dapat diubah.")
    
    # Menampilkan index dari 1
    edited_data = pd.DataFrame({
        'x_value': st.session_state.data['x_values'],
        'y_value': st.session_state.data['y_values']
    })
    edited_data.index = edited_data.index + 1 # Ubah indeks menjadi dari 1
    
    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=False,
        column_config={
            "x_value": st.column_config.NumberColumn("Nilai Tetap (x)", format="%.1f", help="Nilai X ini adalah titik pengukuran standar dan tidak dapat diubah."),
            "y_value": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f", help="Nilai benang putus atau gaya putus dalam Newton (N)"),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes", use_container_width=True):
            try:
                cleaned_edited_df = edited_df.dropna(subset=['x_value', 'y_value'])

                if not np.all(np.diff(cleaned_edited_df['x_value'].values) > 0):
                    st.error("Nilai 'x_value' harus monoton meningkat. Harap perbaiki data Anda.")
                elif cleaned_edited_df.empty:
                    st.warning("Tabel data kosong. Harap masukkan data.")
                else:
                    if len(cleaned_edited_df) != len(INITIAL_DATA['x_values']):
                        st.warning("Jumlah baris data telah berubah. Pastikan Anda hanya mengubah 'Nilai Benang Putus (N)' pada data yang sudah ada atau impor data dengan struktur yang sesuai.")
                        st.session_state.data = pd.DataFrame(INITIAL_DATA)
                        st.session_state.data_needs_recalc = True
                    else:
                        st.session_state.data = pd.DataFrame({
                            'x_values': cleaned_edited_df['x_value'].values,
                            'y_values': cleaned_edited_df['y_value'].values
                        })
                        st.session_state.data_needs_recalc = True
                        st.success("Data berhasil diperbarui! Klik 'Hitung & Tampilkan Grafik' untuk melihat hasilnya.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menerapkan perubahan: {e}. Pastikan data Anda berformat angka.")

    with col2:
        if st.button("Reset Data ke Awal", key="reset_data", use_container_width=True):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.data_needs_recalc = True
            st.success("Data telah direset ke nilai awal.")

with tabs[1]:
    st.write("Unggah file Excel Anda (misalnya `.xlsx`, `.xls`). Pastikan kolom 'x_values' dan 'y_values' ada.")
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"], key="file_uploader")

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            if 'x_values' in df_uploaded.columns and 'y_values' in df_uploaded.columns:
                df_uploaded['x_values'] = pd.to_numeric(df_uploaded['x_values'], errors='coerce')
                df_uploaded['y_values'] = pd.to_numeric(df_uploaded['y_values'], errors='coerce')
                
                df_uploaded.dropna(subset=['x_values', 'y_values'], inplace=True)

                if not np.all(np.diff(df_uploaded['x_values'].values) > 0):
                    st.error("Nilai 'x_values' dari file Excel harus monoton meningkat. Harap perbaiki data Anda.")
                elif df_uploaded.empty:
                    st.warning("File Excel kosong atau tidak mengandung data yang valid setelah pembersihan.")
                else:
                    st.session_state.data = df_uploaded[['x_values', 'y_values']]
                    st.session_state.data_needs_recalc = True
                    st.success("Data dari Excel berhasil diimpor!")
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
            else:
                st.error("File Excel harus mengandung kolom 'x_values' dan 'y_values'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file Excel: {e}. Pastikan format file benar dan kolom sesuai.")

# --- Bagian Analisis & Visualisasi ---
st.subheader("Analisis & Visualisasi")

# Calculate if needed
if st.session_state.data_needs_recalc:
    st.session_state.calculated_results = calculate_lines_and_points(
        st.session_state.data['x_values'],
        st.session_state.data['y_values']
    )
    st.session_state.data_needs_recalc = False

# Pilihan Grafik Analisis
st.subheader("Pilihan Grafik Analisis")
analysis_choice = st.radio(
    "Pilih jenis grafik yang ingin ditampilkan:",
    ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis yang melewati banyak titik", "Tampilkan Semua"),
    key="analysis_choice_radio",
    horizontal=True
)

st.plotly_chart(
    create_abrasion_plot(
        st.session_state.data['x_values'],
        st.session_state.data['y_values'],
        st.session_state.calculated_results,
        analysis_choice
    ),
    use_container_width=True
)

# --- Hasil Perhitungan (Dinamis Berdasarkan Pilihan Grafik) ---
st.subheader("Hasil Perhitungan Titik Potong di X = 50")

if analysis_choice == "Kurva Data Asli":
    st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Kurva Data Asli</h3>", unsafe_allow_html=True)
    if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #FFD700;'>{st.session_state.calculated_results['y_at_x_50_original_curve']:.2f} N</p>", unsafe_allow_html=True)
        st.markdown("<p><i>Interpolasi linear dari kurva data asli pada X=50.</i></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #CCCCCC;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif analysis_choice == "Garis Titik 10 & 20":
    st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Garis Titik 10 & 20</h3>", unsafe_allow_html=True)
    if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #00BFFF;'>{st.session_state.calculated_results['y_at_x_50_pt10_20_line']:.2f} N</p>", unsafe_allow_html=True)
        st.markdown("<p><i>Regresi linear yang melewati titik ke-10 dan ke-20 pada X=50.</i></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #CCCCCC;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif analysis_choice == "Garis yang melewati banyak titik":
    st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Garis RANSAC</h3>", unsafe_allow_html=True)
    if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #32CD32;'>{st.session_state.calculated_results['y_at_x_50_ransac_line']:.2f} N</p>", unsafe_allow_html=True)
        st.markdown("<p><i>Regresi robust RANSAC pada X=50, cocok untuk data dengan outlier.</i></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #CCCCCC;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif analysis_choice == "Tampilkan Semua":
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        with st.container(height=180):
            st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Kurva Data Asli</h3>", unsafe_allow_html=True)
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
                st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #FFD700;'>{st.session_state.calculated_results['y_at_x_50_original_curve']:.2f} N</p>", unsafe_allow_html=True)
                st.markdown("<p><i>Interpolasi linear dari kurva data asli pada X=50.</i></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #CCCCCC;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_res2:
        with st.container(height=180):
            st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Garis Titik 10 & 20</h3>", unsafe_allow_html=True)
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
                st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #00BFFF;'>{st.session_state.calculated_results['y_at_x_50_pt10_20_line']:.2f} N</p>", unsafe_allow_html=True)
                st.markdown("<p><i>Regresi linear yang melewati titik ke-10 dan ke-20.</i></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #CCCCCC;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_res3:
        with st.container(height=180):
            st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Garis RANSAC</h3>", unsafe_allow_html=True)
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
                st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #32CD32;'>{st.session_state.calculated_results['y_at_x_50_ransac_line']:.2f} N</p>", unsafe_allow_html=True)
                st.markdown("<p><i>Regresi robust terhadap semua titik data.</i></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #CCCCCC;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang - Dibuat oleh RADIX
</div>
""", unsafe_allow_html=True)

# Add download section before the footer
st.markdown("---")
st.subheader("Unduh Hasil Analisis")

# Ask for filename
filename = st.text_input("Nama file untuk dokumen Word (tanpa ekstensi .docx)", value="Hasil_Analisis_Abrasi")

# Create download button
if st.button("Unduh Dokumen Word"):
    if not filename:
        st.warning("Silakan masukkan nama file terlebih dahulu")
    else:
        # Create a Word document
        doc = Document()
        doc.add_heading('Hasil Analisis Abrasi Benang', level=1)
        
        # Add date and time
        from datetime import datetime
        doc.add_paragraph(f"Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Section 1: Data
        doc.add_heading('Data Abrasi', level=2)
        doc.add_paragraph('Berikut adalah data abrasi yang digunakan dalam analisis:')
        
        # Create a table for the data
        table = doc.add_table(rows=1, cols=2)
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Nilai X'
        hdr_cells[1].text = 'Nilai Benang Putus (N)'
        
        for x, y in zip(st.session_state.data['x_values'], st.session_state.data['y_values']):
            row_cells = table.add_row().cells
            row_cells[0].text = str(x)
            row_cells[1].text = str(y)
        
        # Section 2: Graph
        doc.add_heading('Grafik Analisis', level=2)
        doc.add_paragraph('Berikut adalah grafik hasil analisis:')
        
        # Save the plot to a temporary file
        fig = create_abrasion_plot(
            st.session_state.data['x_values'],
            st.session_state.data['y_values'],
            st.session_state.calculated_results,
            analysis_choice
        )
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            doc.add_picture(tmpfile.name, width=Inches(6))
        
        # Section 3: Results
        doc.add_heading('Hasil Perhitungan', level=2)
        doc.add_paragraph(f'Nilai perpotongan pada X = {TARGET_X_VALUE}:')
        
        if analysis_choice == "Kurva Data Asli" or analysis_choice == "Tampilkan Semua":
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
                doc.add_paragraph(
                    f"Kurva Data Asli: {st.session_state.calculated_results['y_at_x_50_original_curve']:.2f} N",
                    style='List Bullet'
                )
        
        if analysis_choice == "Garis Titik 10 & 20" or analysis_choice == "Tampilkan Semua":
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
                doc.add_paragraph(
                    f"Garis Titik 10 & 20: {st.session_state.calculated_results['y_at_x_50_pt10_20_line']:.2f} N",
                    style='List Bullet'
                )
        
        if analysis_choice == "Garis yang melewati banyak titik" or analysis_choice == "Tampilkan Semua":
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
                doc.add_paragraph(
                    f"Garis RANSAC: {st.session_state.calculated_results['y_at_x_50_ransac_line']:.2f} N",
                    style='List Bullet'
                )
        
        # Save the document to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
            doc.save(tmp_docx.name)
            
            # Read the file and create download link
            with open(tmp_docx.name, "rb") as f:
                bytes_data = f.read()
                b64 = base64.b64encode(bytes_data).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}.docx">Klik di sini untuk mengunduh</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        st.success("Dokumen Word siap diunduh!")

st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang - Dibuat oleh RADIX
</div>
""", unsafe_allow_html=True)
