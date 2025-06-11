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
    @import url('https://fonts.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap');

    /* Target elemen HTML dan body untuk memastikan background hitam total */
    html, body {
        background-color: #0A0A0A !important;
        color: #E0E0E0; /* Pastikan teks juga terang */
        /* overflow-x: hidden; --- Dihapus untuk fleksibilitas scroll horizontal jika dibutuhkan */
    }

    /* Streamlit's main wrapper */
    .stApp {
        background-color: #0A0A0A !important; /* Background untuk seluruh aplikasi Streamlit */
        max-width: 1300px; /* Lebar maksimal untuk desktop */
        margin: 0 auto;
        padding-top: 30px; /* Padding atas lebih besar */
        padding-bottom: 50px; /* Padding bawah untuk footer */
        padding-left: 15px; /* Padding samping default */
        padding-right: 15px; /* Padding samping default */
    }
    
    /* Main content area within .stApp */
    .main {
        background-color: #0A0A0A; /* Lebih gelap dari #121212 */
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif; /* Font umum yang lebih modern */
    }
    /* Kontainer utama untuk sidebar jika ada (biasanya di desktop, tapi bisa muncul di mobile) */
    .stSidebar {
        background-color: #0A0A0A !important; /* Jika ada sidebar, pastikan juga hitam */
        color: #E0E0E0;
    }
    .block-container {
        background-color: #0A0A0A !important; /* Kontainer blok utama Streamlit */
        padding-top: 1rem; /* Kurangi padding atas untuk mobile */
        padding-bottom: 1rem; /* Kurangi padding bawah untuk mobile */
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #F8F8F8; /* Sedikit lebih putih dari #FFFFFF */
        font-family: 'Playfair Display', serif; /* Font serif untuk judul, kesan mewah */
        letter-spacing: 0.8px; /* Jarak huruf lebih lebar */
        word-break: break-word; /* Memastikan teks panjang tidak meluber */
    }
    h1 {
        font-weight: 700;
        font-size: 44px; /* Ukuran desktop */
        padding-bottom: 15px;
        border-bottom: 3px solid #8B4513;
        text-align: center;
        text-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    h2 {
        font-weight: 600;
        font-size: 32px; /* Ukuran desktop */
        color: #DAA520;
        margin-bottom: 20px;
        border-bottom: 1px solid #282828;
        padding-bottom: 8px;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 0.5px;
    }
    h3 {
        font-weight: 600;
        font-size: 24px; /* Ukuran desktop */
        color: #F8F8F8;
        font-family: 'Montserrat', sans-serif;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    p, li, span, div {
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif;
        line-height: 1.8;
        font-size: 17px; /* Ukuran desktop, sedikit lebih besar dari sebelumnya */
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #8B4513;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 12px 25px; /* Ukuran desktop */
        font-size: 17px; /* Ukuran desktop */
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        white-space: normal; /* Memungkinkan teks tombol wrap */
        word-break: break-word;
    }
    .stButton>button:hover {
        background-color: #A0522D;
        box_shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
        transform: translateY(-3px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Montserrat', sans-serif;
        color: #B0B0B0;
        font-weight: 600;
        padding: 12px 20px; /* Ukuran desktop */
        font-size: 17px; /* Ukuran desktop */
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 12px;
        background-color: #1A1A1A;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        border: 1px solid #282828;
        overflow-x: auto; /* Memungkinkan tab discroll horizontal jika banyak */
        -webkit-overflow-scrolling: touch; /* Untuk scrolling yang mulus di iOS */
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 30px; /* Ukuran desktop */
        border-radius: 12px;
        background-color: #1A1A1A;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #DAA520;
        border-radius: 6px;
        height: 4px;
    }

    /* Radio Buttons - Unified for Graph & Results */
    .stRadio > label {
        color: #F8F8F8;
        font-size: 18px; /* Ukuran desktop */
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stRadio > div { /* Container for radio buttons */
        background-color: #1A1A1A;
        border-radius: 12px;
        padding: 20px; /* Ukuran desktop */
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        display: flex;
        flex-wrap: wrap; /* Memungkinkan item wrap ke baris baru */
        gap: 15px;
        justify-content: center; /* Pusatkan opsi radio */
    }
    .stRadio [data-baseweb="radio"] { /* Individual radio item */
        background-color: #282828;
        border-radius: 10px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, border 0.3s ease;
        flex-grow: 1; /* Memungkinkan item tumbuh mengisi ruang */
        text-align: center;
        min-width: 150px; /* Minimal lebar untuk setiap opsi */
    }
    .stRadio [data-baseweb="radio"]:hover {
        background-color: #3A3A3A;
        border: 1px solid #DAA520;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: #DAA520 !important;
        color: white;
        border: 1px solid #DAA520;
        box-shadow: 0 4px 15px rgba(218, 165, 32, 0.4);
    }
    .stRadio [data-baseweb="radio"] span:last-child { /* text of the radio button */
        color: #E0E0E0;
        font-weight: 600;
        font-size: 17px; /* Ukuran desktop */
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: white;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1A1A1A;
        border-radius: 15px;
        padding: 30px; /* Ukuran desktop */
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        margin-bottom: 30px;
        border: 1px solid #282828;
    }

    /* Radix Header (PULCRA Branding) */
    .app-header {
        background: linear-gradient(145deg, #1A1A1A, #0A0A0A);
        padding: 40px; /* Ukuran desktop */
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(8px);
        border: 1px solid #282828;
    }
    .pulcra-logo {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 56px; /* Ukuran desktop */
        color: #DAA520;
        margin-bottom: 10px;
        letter-spacing: 5px;
        text-shadow: 0 5px 20px rgba(218, 165, 32, 0.5);
        text-transform: uppercase;
    }
    .app-header h1 {
        font-size: 38px; /* Ukuran desktop */
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0;
        text-shadow: none;
        color: #F8F8F8;
    }
    .app-header p {
        font-size: 18px; /* Ukuran desktop */
        color: #B0B0B0;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }

    /* Footer */
    .radix-footer {
        text-align: center;
        margin-top: 60px;
        padding: 25px; /* Ukuran desktop */
        font-size: 15px;
        font-family: 'Montserrat', sans-serif;
        color: #A0A0A0;
        border-top: 1px solid #282828;
        background-color: #1A1A1A;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.3);
    }

    /* Other elements */
    hr {
        border-color: #282828 !important;
        margin: 40px 0 !important;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    
    /* Streamlit specific adjustments for better dark mode */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0px !important;
        position: fixed !important;
    }
    .stApp:hover [data-testid="stToolbar"] {
        visibility: visible !important;
        height: auto !important;
    }

    /* For data editor and file uploader to blend better */
    [data-testid="stDataEditor"] {
        border-radius: 10px;
        overflow: auto; /* Penting untuk scroll horizontal di mobile */
        border: 1px solid #282828;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #DAA520;
        border-radius: 12px;
        padding: 25px; /* Ukuran desktop */
        background-color: #1A1A1A;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploaderDropzone"] p {
        color: #B0B0B0;
        font-size: 17px; /* Ukuran desktop */
    }

    /* For data preview tables */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.3);
        background-color: #1A1A1A;
        max-height: 350px;
        overflow-y: auto;
        overflow-x: auto; /* Sangat penting untuk tabel di mobile */
        border: 1px solid #282828;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
        min-width: 600px; /* Pastikan tabel punya lebar minimum untuk scroll */
    }
    .stDataFrame th {
        background-color: #282828 !important;
        color: #DAA520 !important;
        font-weight: 700;
        position: sticky;
        top: 0;
        z-index: 1;
        font-size: 16px; /* Ukuran desktop */
    }
    .stDataFrame td {
        background-color: #1A1A1A !important;
        color: #E0E0E0 !important;
        border-bottom: 1px solid #282828 !important;
        padding: 10px 15px;
    }
    /* Scrollbar for dataframes */
    .stDataFrame::-webkit-scrollbar {
        width: 10px;
        height: 10px; /* Untuk scrollbar horizontal */
    }
    .stDataFrame::-webkit-scrollbar-track {
        background: #1A1A1A;
    }
    .stDataFrame::-webkit-scrollbar-thumb {
        background: #DAA520;
        border-radius: 10px;
    }
    .stDataFrame::-webkit-scrollbar-thumb:hover {
        background: #C49F3D;
    }

    /* Plotly specifics for dark elegance */
    .js-plotly-plot .plotly .modebar {
        background-color: #1A1A1A !important;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .js-plotly-plot .plotly .modebar-btn {
        color: #DAA520 !important;
    }
    .js-plotly-plot .plotly .modebar-btn:hover {
        background-color: #282828 !important;
    }

    /* Access Code styling */
    .stTextInput>div>div>input {
        background-color: #1A1A1A;
        border: 1px solid #282828;
        border-radius: 8px;
        color: #E0E0E0;
        padding: 10px 15px;
        font-size: 18px; /* Ukuran desktop */
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput>label {
        font-size: 18px; /* Ukuran desktop */
        color: #F8F8F8;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .st-emotion-cache-16txt4s { /* Ini adalah selector untuk error message Streamlit */
        background-color: #4A0000;
        color: #FFCCCC;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #8B0000;
    }
    .st-emotion-cache-zt5ig8 { /* Ini adalah selector untuk success message Streamlit */
        background-color: #004A00;
        color: #CCFFCC;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #008B00;
    }

    /* --- MEDIA QUERIES FOR MOBILE RESPONSIVENESS --- */
    @media (max-width: 768px) {
        .stApp {
            padding-left: 10px; /* Kurangi padding samping untuk layar kecil */
            padding-right: 10px; /* Kurangi padding samping untuk layar kecil */
            padding-top: 20px; /* Kurangi padding atas */
            padding-bottom: 30px; /* Kurangi padding bawah */
        }
        h1 {
            font-size: 32px; /* Perkecil ukuran h1 untuk mobile */
            padding-bottom: 10px;
        }
        h2 {
            font-size: 24px; /* Perkecil ukuran h2 untuk mobile */
            margin-bottom: 15px;
        }
        h3 {
            font-size: 20px; /* Perkecil ukuran h3 untuk mobile */
            margin-top: 20px;
            margin-bottom: 10px;
        }
        p, li, span, div {
            font-size: 15px; /* Perkecil ukuran font teks biasa, disesuaikan */
            line-height: 1.6;
        }
        .stButton>button {
            padding: 10px 20px; /* Perkecil padding tombol */
            font-size: 15px; /* Perkecil font tombol */
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 15px; /* Perkecil padding tab */
            font-size: 15px; /* Perkecil font tab */
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding: 20px; /* Perkecil padding panel tab */
        }
        .stRadio > label {
            font-size: 16px; /* Perkecil font label radio */
            margin-bottom: 10px;
        }
        .stRadio [data-baseweb="radio"] {
            min-width: unset; /* Hapus min-width agar lebih fleksibel */
            padding: 8px 15px; /* Perkecil padding item radio */
            font-size: 15px; /* Perkecil font item radio */
        }
        .app-header {
            padding: 30px; /* Perkecil padding header */
            margin-bottom: 30px;
        }
        .pulcra-logo {
            font-size: 40px; /* Perkecil logo */
            letter-spacing: 3px;
        }
        .app-header h1 {
            font-size: 28px; /* Perkecil h1 di header */
        }
        .app-header p {
            font-size: 16px; /* Perkecil font di header */
        }
        .dark-card {
            padding: 20px; /* Perkecil padding card */
            margin-bottom: 20px;
        }
        .stDataFrame th, .stDataFrame td {
            font-size: 14px; /* Perkecil font tabel */
            padding: 8px 10px;
        }
        .stTextInput>div>div>input {
            font-size: 16px; /* Perkecil font input teks */
            padding: 8px 12px;
        }
        .stTextInput>label {
            font-size: 16px; /* Perkecil font label input */
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
            font-size: 14px; /* Penyesuaian lebih lanjut untuk mobile kecil */
        }
        .stButton>button {
            padding: 8px 15px;
            font-size: 14px;
        }
        .stRadio > div {
            flex-direction: column; /* Tumpuk radio button secara vertikal */
            align-items: stretch; /* Regangkan item radio */
        }
        .stRadio [data-baseweb="radio"] {
            width: 100%; /* Lebar penuh */
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
if 'update_graph' not in st.session_state:
    st.session_state.update_graph = False
if 'calculated_results' not in st.session_state:
    st.session_state.calculated_results = {}
if 'password_entered' not in st.session_state:
    st.session_state.password_entered = False
if 'custom_line_params' not in st.session_state:
    st.session_state.custom_line_params = {'x1': 0.0, 'y1': 0.0, 'x2': 100.0, 'y2': 1000.0} # Default values
if 'custom_line_intersection' not in st.session_state:
    st.session_state.custom_line_intersection = np.nan # Inisialisasi dengan NaN
if 'custom_points_clicked' not in st.session_state:
    st.session_state.custom_points_clicked = [] # List untuk menyimpan koordinat klik

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
        return results

    # Original curve interpolation
    try:
        f = interpolate.interp1d(x_np, y_np, kind='linear', fill_value='extrapolate')
        results['y_at_x_50_original_curve'] = float(f(50))
    except ValueError:
        st.warning("Tidak dapat melakukan interpolasi kurva asli. Periksa data X Anda (harus monoton meningkat).")
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
            results['y_at_x_50_pt10_20_line'] = slope_pt10_20 * 50 + intercept_pt10_20

            x_min_plot = x_np.min() if x_np.size > 0 else 0
            x_max_plot = x_np.max() if x_np.size > 0 else 100
            results['pt10_20_line_x_range'] = np.linspace(min(x_min_plot, 50), max(x_max_plot, 50), 100)
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
            residual_threshold_val = np.std(y_np) * 0.5 
            if len(y_np) <= 1 or np.std(y_np) == 0:
                residual_threshold_val = 1.0 # Default fallback jika std dev nol atau tidak cukup data

            ransac = RANSACRegressor(LinearRegression(),
                                     min_samples=2,
                                     residual_threshold=residual_threshold_val,
                                     random_state=42,
                                     max_trials=1000)
            ransac.fit(X_reshaped, y_np)
            results['y_at_x_50_ransac_line'] = ransac.predict(np.array([[50]]))[0]

            x_min_plot = x_np.min() if x_np.size > 0 else 0
            x_max_plot = x_np.max() if x_np.size > 0 else 100
            
            results['ransac_line_x'] = np.linspace(min(x_min_plot, 50), max(x_max_plot, 50), 100)
            results['ransac_line_y'] = ransac.predict(results['ransac_line_x'].reshape(-1, 1))
            
        except Exception as e:
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])
            results['y_at_x_50_ransac_line'] = np.nan

    return results

# --- Fungsi untuk Menghitung Perpotongan Garis Kustom dengan X=50 ---
def calculate_custom_line_intersection(x1, y1, x2, y2):
    # Cek jika x1, y1, x2, y2 adalah NaN atau None (jika belum diisi)
    if any(np.isnan([x1, y1, x2, y2])) or x1 is None or y1 is None or x2 is None or y2 is None:
        return np.nan

    if x1 == x2: # Garis vertikal
        if x1 == 50:
            return y1 # Jika garis vertikal tepat di x=50, ambil y1 sebagai titik potong (atau y2, sama saja)
        else:
            return np.nan # Tidak berpotongan dengan x=50 jika bukan x=50
    
    # Hitung persamaan garis y = mx + c
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    
    # Perpotongan dengan x=50
    y_intersect = m * 50 + c
    
    # Periksa apakah titik perpotongan berada dalam segmen garis (x1,x2)
    # Menggunakan toleransi kecil untuk floating point comparison
    tolerance = 1e-9
    if not (min(x1, x2) - tolerance <= 50 <= max(x1, x2) + tolerance):
        return np.nan # x=50 di luar segmen garis
        
    return y_intersect

# --- Bagian Input Data ---
st.subheader("Input Data")
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.write("Ubah nilai Y (N atau nilai benang putus) dari data abrasi. Nilai X tetap.")
    
    # Menampilkan index dari 1
    edited_data = pd.DataFrame({
        'x_value': st.session_state.data['x_values'],
        'y_value': st.session_state.data['y_values']
    })
    edited_data.index = edited_data.index + 1 # Ubah indeks menjadi dari 1
    
    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=False, # Tampilkan indeks
        column_config={
            "x_value": st.column_config.NumberColumn("Nilai Tetap (x)", format="%.1f"),
            "y_value": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f", help="Nilai benang putus atau gaya putus dalam Newton (N)"),
        },
        use_container_width=True,
        key="data_editor",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes", use_container_width=True):
            try:
                # Ambil data tanpa index yang sudah dimodifikasi
                new_y_values = edited_df['y_value'].astype(float).tolist()
                # Pastikan jumlah X dan Y tetap sama setelah editan
                if len(new_y_values) == len(st.session_state.data['x_values']):
                    st.session_state.data['y_values'] = new_y_values
                    st.session_state.update_graph = True 
                    calculate_lines_and_points.clear() # Clear cache
                    st.success("Data berhasil diperbarui!")
                else:
                    st.error("Jumlah baris Y tidak boleh berubah. Silakan sesuaikan atau tambahkan baris jika diperlukan.")
            except ValueError:
                st.error("Pastikan semua nilai Y adalah angka yang valid.")
    
    with col2:
        if st.button("Reset Data Awal", key="reset_values", use_container_width=True):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.update_graph = True 
            calculate_lines_and_points.clear() # Clear cache
            st.success("Data berhasil direset ke nilai awal!")

with tabs[1]:
    st.write("Unggah file Excel dengan kolom **'x_values'** dan **'y_values'**.")
    # Pindahkan tombol unduh template ke sini agar lebih relevan
    if st.button("Unduh Template Excel", use_container_width=True, key="download_template_btn"):
        sample_df = pd.DataFrame(INITIAL_DATA)
        buffer = io.BytesIO()
        sample_df.to_excel(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Klik untuk Mengunduh Template",
            data=buffer,
            file_name="template_abrasi_benang.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_button_actual" # Unique key for the actual download button
        )
    
    uploaded_file = st.file_uploader("Pilih File Excel Anda", type=['xlsx', 'xls'], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            if 'x_values' in df_uploaded.columns and 'y_values' in df_uploaded.columns:
                st.write("Pratinjau Data Impor:")
                # Tampilkan data dengan indeks dari 1 dan bisa di-scroll
                df_uploaded_display = df_uploaded.copy()
                df_uploaded_display.index = df_uploaded_display.index + 1
                st.dataframe(
                    df_uploaded_display,
                    use_container_width=True,
                    height=300, # Atur tinggi agar bisa di-scroll
                    column_config={
                        "x_values": st.column_config.NumberColumn("Nilai Tetap (x)", format="%.1f"),
                        "y_values": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f"),
                    },
                    key="uploaded_df_preview"
                )
                
                # Periksa apakah data valid sebelum tombol "Gunakan Data Ini"
                x_check = df_uploaded['x_values'].astype(float).dropna()
                y_check = df_uploaded['y_values'].astype(float).dropna()

                if len(x_check) != len(y_check):
                    st.error("Jumlah nilai X dan Y harus sama. Periksa file Excel Anda.")
                elif x_check.empty:
                    st.warning("File Excel tidak memiliki data yang valid setelah pembersihan. Periksa formatnya.")
                else:
                    if st.button("Gunakan Data Ini", key="use_imported", use_container_width=True):
                        st.session_state.data['x_values'] = x_check.tolist()
                        st.session_state.data['y_values'] = y_check.tolist()
                        st.session_state.update_graph = True
                        calculate_lines_and_points.clear() # Clear cache
                        st.success("Data impor berhasil diterapkan!")
            else:
                st.error("File Excel harus mengandung kolom **'x_values'** dan **'y_values'**. Harap periksa nama kolom Anda.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file Excel: {e}. Pastikan format file dan data benar.")

st.markdown("---") # Garis pemisah

# --- Bagian Grafik dan Hasil ---

st.subheader("Visualisasi & Hasil Analisis")

# Perbarui perhitungan jika data berubah
if st.session_state.update_graph or not st.session_state.calculated_results:
    x_values_current = pd.Series(st.session_state.data['x_values'])
    y_values_current = pd.Series(st.session_state.data['y_values'])
    
    if len(x_values_current) < 2 or len(y_values_current) < 2:
        st.warning("Data tidak cukup untuk analisis. Masukkan minimal 2 pasangan X dan Y.")
        # Clear results if data is insufficient to avoid displaying stale results
        st.session_state.calculated_results = {}
    else:
        st.session_state.calculated_results = calculate_lines_and_points(x_values_current, y_values_current)
    st.session_state.update_graph = False

results = st.session_state.calculated_results
x_values = pd.Series(st.session_state.data['x_values'])
y_values = pd.Series(st.session_state.data['y_values'])

# UNIFIED RADIO BUTTON UNTUK GRAFIK DAN HASIL 
analysis_choice = st.radio(
    "Pilih jenis analisis yang ingin ditampilkan:",
    ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis Regresi RANSAC", "Garis Kustom", "Tampilkan Semua"),
    key="analysis_choice",
    horizontal=True
)

st.markdown("---") 
st.write("#### Grafik Abrasi Benang") 

fig = go.Figure()

# Tambahkan Kurva Data Abrasi (selalu) 
if not x_values.empty and not y_values.empty:
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values,
        mode='lines+markers',
        name='Data Abrasi',
        line=dict(color='#8B4513', width=3), # Warna emas/tembaga untuk kurva asli
        marker=dict(size=8, color='#DAA520') # Emas gelap untuk marker
    ))

    # Tambahkan Garis Vertikal di x=50 (selalu) 
    min_y_for_line = y_values.min() if not y_values.empty else 0
    max_y_for_line = y_values.max() if not y_values.empty else 100
    
    # Perbaiki rentang y0 dan y1 untuk garis vertikal agar tidak terlalu jauh atau terlalu dekat
    # Sesuaikan rentang plot y agar garis vertikal selalu terlihat.
    # Jika y_values ada, ambil min/max y, jika tidak, pakai default 0-1000
    if not y_values.empty:
        plot_y_min = y_values.min()
        plot_y_max = y_values.max()
        y_range_span = plot_y_max - plot_y_min
        y0_line = plot_y_min - y_range_span * 0.1 # Sedikit di bawah min data
        y1_line = plot_y_max + y_range_span * 0.1 # Sedikit di atas max data
    else:
        y0_line = 0
        y1_line = 1000

    fig.add_shape(
        type="line",
        x0=50, y0=y0_line,
        x1=50, y1=y1_line,
        line=dict(color="#FF4500", width=2, dash="dash"), # Oranye kemerahan yang kuat
    )
    # Sesuaikan posisi anotasi x=50 agar tidak tumpang tindih
    fig.add_annotation(
        x=50, y=y1_line * 0.95, # Posisi di dekat puncak garis vertikal
        text="x=50", showarrow=False,
        font=dict(color="#FF4500", size=14, family="Montserrat, sans-serif", weight="bold"),
        bgcolor="rgba(26,26,26,0.7)", bordercolor="#FF4500", borderwidth=1, borderpad=4
    )

    # Tambahkan titik perpotongan kurva asli dengan x=50 (selalu) 
    if not np.isnan(results.get('y_at_x_50_original_curve', np.nan)):
        fig.add_trace(go.Scatter(
            x=[50], y=[results['y_at_x_50_original_curve']],
            mode='markers',
            name=f'Int. Kurva Asli di x=50, y={results["y_at_x_50_original_curve"]:.2f}',
            marker=dict(size=14, color='#FF4500', symbol='circle', line=dict(width=2, color='white'))
        ))

# Kondisional untuk Garis Titik 10 & 20 
if analysis_choice in ["Garis Titik 10 & 20", "Tampilkan Semua"]:
    if not np.isnan(results.get('specific_x1_pt10_20', np.nan)) and not np.isnan(results.get('specific_x2_pt10_20', np.nan)):
        # Hanya tambahkan titik referensi jika data cukup untuk titik 10 dan 20 spesifik
        if len(x_values) >= 20:
             fig.add_trace(go.Scatter(
                x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
                y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
                mode='markers', name='Titik Referensi (10 & 20)',
                marker=dict(size=12, color='#FFD700', symbol='star', line=dict(width=2, color='white'))
            ))
        elif len(x_values) >= 2:
            fig.add_trace(go.Scatter(
                x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
                y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
                mode='markers', name='Titik Referensi (Pertama & Terakhir)', # Ubah nama legend
                marker=dict(size=12, color='#FFD700', symbol='star', line=dict(width=2, color='white'))
            ))
            st.info("Catatan: Dataset Anda kurang dari 20 titik. Garis 'Titik 10 & 20' dihitung antara titik pertama dan terakhir data Anda saat ini.")

        if results['pt10_20_line_x_range'].size > 0: # Pastikan ada data untuk garis
            fig.add_trace(go.Scatter(
                x=results['pt10_20_line_x_range'], y=results['pt10_20_line_y'],
                mode='lines', name='Garis Titik 10 & 20',
                line=dict(color="#B8860B", width=3, dash="dot"), showlegend=True
            ))
            if not np.isnan(results.get('y_at_x_50_pt10_20_line', np.nan)):
                # Posisi label agar tidak tumpang tindih
                y_pos_pt10_20_label = results['y_at_x_50_pt10_20_line'] + (y_range_span * 0.05 if y_range_span > 0 else 50)
                fig.add_trace(go.Scatter(
                    x=[50], y=[results['y_at_x_50_pt10_20_line']],
                    mode='markers', name=f'Int. Garis 10-20 di x=50, y={results["y_at_x_50_pt10_20_line"]:.2f}',
                    marker=dict(size=14, color='#B8860B', symbol='circle-open', line=dict(width=3, color='#B8860B'))
                ))
                fig.add_annotation(
                    x=50, y=y_pos_pt10_20_label, text=f"Garis 10-20: {results['y_at_x_50_pt10_20_line']:.2f}",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#B8860B',
                    font=dict(size=14, color='#B8860B', family="Montserrat, sans-serif"),
                    bordercolor="#B8860B", borderwidth=1, borderpad=4, bgcolor="rgba(26,26,26,0.7)", opacity=0.9
                )

# Kondisional untuk Garis Regresi RANSAC 
if analysis_choice in ["Garis Regresi RANSAC", "Tampilkan Semua"]:
    if results['ransac_line_x'].size > 0 and not np.isnan(results.get('y_at_x_50_ransac_line', np.nan)):
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'], y=results['ransac_line_y'],
            mode='lines', name='Regresi RANSAC',
            line=dict(color='#00CED1', width=3, dash='dash'), showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[50], y=[results['y_at_x_50_ransac_line']],
            mode='markers', name=f'Int. RANSAC di x=50, y={results["y_at_x_50_ransac_line"]:.2f}',
            marker=dict(size=14, color='#00CED1', symbol='diamond-open', line=dict(width=3, color='#00CED1'))
        ))
        # Posisi label agar tidak tumpang tindih
        y_pos_ransac_label = results['y_at_x_50_ransac_line'] - (y_range_span * 0.05 if y_range_span > 0 else 50) 
        if y_pos_ransac_label < y0_line: # Pastikan label tidak keluar dari batas bawah plot
            y_pos_ransac_label = y0_line + (y_range_span * 0.02 if y_range_span > 0 else 5) # Sedikit di atas batas bawah
            
        fig.add_annotation(
            x=50, y=y_pos_ransac_label, text=f"RANSAC: {results['y_at_x_50_ransac_line']:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#00CED1',
            font=dict(size=14, color='#00CED1', family="Montserrat, sans-serif"),
            bordercolor="#00CED1", borderwidth=1, borderpad=4, bgcolor="rgba(26,26,26,0.7)", opacity=0.9
        )
    elif analysis_choice in ["Garis Regresi RANSAC", "Tampilkan Semua"]:
        if len(x_values) < 2:
            st.warning("Tidak cukup data untuk menghitung Regresi RANSAC (minimal 2 titik).")
        else:
            st.warning("Regresi RANSAC tidak dapat dihitung dengan data yang diberikan. Coba periksa outlier atau distribusi data.")

# --- Bagian Input Garis Kustom (Modifikasi untuk Klik Manual) ---
if analysis_choice in ["Garis Kustom", "Tampilkan Semua"]:
    st.markdown("#### Gambar Garis Kustom (Input Manual Koordinat)")
    st.info("Untuk menggambar garis kustom, Anda dapat memasukkan dua titik (X1, Y1) dan (X2, Y2) secara manual. "
            "Gunakan fitur _hover_ pada grafik di atas untuk melihat koordinat X dan Y yang ingin Anda gunakan.")

    col_x1, col_y1, col_x2, col_y2 = st.columns(4)
    with col_x1:
        st.session_state.custom_line_params['x1'] = st.number_input("X1", value=float(st.session_state.custom_line_params['x1']), format="%.1f", key="custom_x1")
    with col_y1:
        st.session_state.custom_line_params['y1'] = st.number_input("Y1", value=float(st.session_state.custom_line_params['y1']), format="%.1f", key="custom_y1")
    with col_x2:
        st.session_state.custom_line_params['x2'] = st.number_input("X2", value=float(st.session_state.custom_line_params['x2']), format="%.1f", key="custom_x2")
    with col_y2:
        st.session_state.custom_line_params['y2'] = st.number_input("Y2", value=float(st.session_state.custom_line_params['y2']), format="%.1f", key="custom_y2")

    # Tombol untuk mereset input garis kustom
    if st.button("Reset Garis Kustom", key="reset_custom_line", use_container_width=True):
        st.session_state.custom_line_params = {'x1': 0.0, 'y1': 0.0, 'x2': 100.0, 'y2': 1000.0}
        st.session_state.custom_line_intersection = np.nan
        st.rerun() # Rerun untuk membersihkan input dan grafik

    # Hitung dan tampilkan garis kustom
    x1, y1 = st.session_state.custom_line_params['x1'], st.session_state.custom_line_params['y1']
    x2, y2 = st.session_state.custom_line_params['x2'], st.session_state.custom_line_params['y2']

    # Update intersection calculation whenever custom line params change
    st.session_state.custom_line_intersection = calculate_custom_line_intersection(x1, y1, x2, y2)

    # Pastikan x1 dan x2 tidak sama untuk menghindari pembagian nol pada kemiringan
    if x1 != x2:
        
        # Ekstrak rentang x dari data asli untuk menentukan panjang garis kustom
        x_min_data = x_values.min() if not x_values.empty else 0
        x_max_data = x_values.max() if not x_values.empty else 100

        # Gunakan rentang yang lebih luas untuk memastikan garis kustom terlihat
        x_range_line = np.linspace(min(x1, x2, x_min_data, 50), max(x1, x2, x_max_data, 50), 100)
        
        # Hitung y untuk rentang x tersebut
        m_custom = (y2 - y1) / (x2 - x1)
        c_custom = y1 - m_custom * x1
        y_range_line = m_custom * x_range_line + c_custom

        fig.add_trace(go.Scatter(
            x=x_range_line, y=y_range_line,
            mode='lines', name='Garis Kustom',
            line=dict(color='#8A2BE2', width=3, dash='longdashdot'), # Warna ungu
            showlegend=True
        ))
        
        # Tambahkan marker untuk titik input kustom
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='markers', name='Titik Input Kustom',
            marker=dict(size=12, color='#8A2BE2', symbol='cross', line=dict(width=2, color='white'))
        ))

        if not np.isnan(st.session_state.custom_line_intersection):
            fig.add_trace(go.Scatter(
                x=[50], y=[st.session_state.custom_line_intersection],
                mode='markers', name=f'Int. Kustom di x=50, y={st.session_state.custom_line_intersection:.2f}',
                marker=dict(size=14, color='#8A2BE2', symbol='square-open', line=dict(width=3, color='#8A2BE2'))
            ))
            # Posisi label agar tidak tumpang tindih
            y_pos_custom_label = st.session_state.custom_line_intersection + (y_range_span * 0.05 if y_range_span > 0 else 50)
            if y_pos_custom_label > y1_line: # Jangan sampai label keluar dari batas atas plot
                y_pos_custom_label = y1_line * 0.98 # Sedikit di bawah batas atas
            
            fig.add_annotation(
                x=50, y=y_pos_custom_label, text=f"Kustom: {st.session_state.custom_line_intersection:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#8A2BE2',
                font=dict(size=14, color='#8A2BE2', family="Montserrat, sans-serif"),
                bordercolor="#8A2BE2", borderwidth=1, borderpad=4, bgcolor="rgba(26,26,26,0.7)", opacity=0.9
            )
        else:
            st.warning("Garis kustom tidak berpotongan dengan x=50 dalam segmen yang ditentukan atau garis vertikal tepat di x=50.")
    else:
        # Tambahan penanganan jika x1 == x2 (garis vertikal)
        if x1 == 50:
            st.warning("Garis kustom adalah garis vertikal tepat di x=50. Titik potong Y akan diambil dari Y1.")
        else:
            st.warning("Garis kustom adalah garis vertikal (X1 sama dengan X2) dan tidak berpotongan dengan garis X=50 kecuali jika X1=50.")
        st.session_state.custom_line_intersection = calculate_custom_line_intersection(x1, y1, x2, y2) # Recalculate to ensure NaN if not 50

# Update layout Plotly
fig.update_layout(
    title_text='Grafik Abrasi Benang vs. Nilai Putus',
    title_x=0.5,
    xaxis_title='Nilai Tetap (x)',
    yaxis_title='Nilai Benang Putus (N)',
    plot_bgcolor='#1A1A1A', # Background plot
    paper_bgcolor='#1A1A1A', # Background area figure
    font=dict(color='#E0E0E0', family='Montserrat, sans-serif'),
    xaxis=dict(gridcolor='#282828', zerolinecolor='#282828'),
    yaxis=dict(gridcolor='#282828', zerolinecolor='#282828'),
    hovermode='x unified', # Mode hover untuk menampilkan koordinat
    margin=dict(l=50, r=50, t=80, b=50),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(26,26,26,0.8)",
        bordercolor="#282828",
        borderwidth=1,
        font=dict(size=12)
    )
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})


st.markdown("---")

# Bagian Hasil Perhitungan
st.write("#### Hasil Perhitungan Perpotongan di X=50")

# Gunakan card untuk menampilkan hasil
col_res1, col_res2, col_res3, col_res4 = st.columns(4) # Menambah kolom untuk garis kustom

with col_res1:
    with st.container(height=150): # Menggunakan container untuk ukuran yang konsisten
        st.markdown(f"""
        <div class="dark-card" style="padding: 15px; text-align: center; height: 100%;">
            <h3 style="font-size: 18px; margin-top: 0; margin-bottom: 5px; color: #DAA520;">Kurva Data Asli</h3>
            <p style="font-size: 24px; font-weight: 700; color: #F8F8F8;">{results.get('y_at_x_50_original_curve', np.nan):.2f}</p>
            <p style="font-size: 12px; color: #B0B0B0;">Nilai Y pada X=50</p>
        </div>
        """, unsafe_allow_html=True)

with col_res2:
    with st.container(height=150):
        st.markdown(f"""
        <div class="dark-card" style="padding: 15px; text-align: center; height: 100%;">
            <h3 style="font-size: 18px; margin-top: 0; margin-bottom: 5px; color: #B8860B;">Garis Titik 10 & 20</h3>
            <p style="font-size: 24px; font-weight: 700; color: #F8F8F8;">{results.get('y_at_x_50_pt10_20_line', np.nan):.2f}</p>
            <p style="font-size: 12px; color: #B0B0B0;">Nilai Y pada X=50</p>
        </div>
        """, unsafe_allow_html=True)

with col_res3:
    with st.container(height=150):
        st.markdown(f"""
        <div class="dark-card" style="padding: 15px; text-align: center; height: 100%;">
            <h3 style="font-size: 18px; margin-top: 0; margin-bottom: 5px; color: #00CED1;">Garis Regresi RANSAC</h3>
            <p style="font-size: 24px; font-weight: 700; color: #F8F8F8;">{results.get('y_at_x_50_ransac_line', np.nan):.2f}</p>
            <p style="font-size: 12px; color: #B0B0B0;">Nilai Y pada X=50</p>
        </div>
        """, unsafe_allow_html=True)

with col_res4:
    with st.container(height=150):
        # Perbaikan utama ada di sini: menggunakan f-string kondisional
        display_custom_intersection = (
            f"{st.session_state.custom_line_intersection:.2f}"
            if not np.isnan(st.session_state.custom_line_intersection)
            else "N/A"
        )
        st.markdown(f"""
        <div class="dark-card" style="padding: 15px; text-align: center; height: 100%;">
            <h3 style="font-size: 18px; margin-top: 0; margin-bottom: 5px; color: #8A2BE2;">Garis Kustom</h3>
            <p style="font-size: 24px; font-weight: 700; color: #F8F8F8;">{display_custom_intersection}</p>
            <p style="font-size: 12px; color: #B0B0B0;">Nilai Y pada X=50</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- Footer ---
st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang &copy; 2025 PULCRA by Radix.
</div>
""", unsafe_allow_html=True)
