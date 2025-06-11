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
        color: #E0E0E0; /* Pastikan teks juga terang */
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
        box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
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
    /* Khusus untuk card hasil, pastikan tidak ada scrollbar */
    .dark-card.result-card {
        height: auto !important; /* Penting: Pastikan tinggi menyesuaikan konten */
        overflow: visible !important; /* Penting: Pastikan tidak ada scrollbar */
        display: flex; /* Gunakan flexbox untuk penataan konten internal */
        flex-direction: column;
        justify-content: center; /* Pusatkan konten vertikal */
        align-items: center; /* Pusatkan konten horizontal */
        text-align: center; /* Pusatkan teks */
        padding: 25px; /* Sedikit kurangi padding agar lebih ringkas */
    }
    .dark-card.result-card h3 {
        margin-bottom: 10px; /* Tambah sedikit jarak antara judul dan nilai */
        text-align: center; /* Pastikan judul di tengah */
    }
    .dark-card.result-card p {
        text-align: center; /* Pastikan paragraf juga di tengah */
    }
    .dark-card.result-card p:last-child {
        margin-top: 10px; /* Tambah sedikit jarak antara nilai dan keterangan */
        font-size: 13px; /* Sedikit perkecil font keterangan */
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
        .stRadio > div {
            flex-direction: column; /* Ubah ke kolom untuk mobile */
            align-items: stretch; /* Regangkan item agar mengisi lebar */
            padding: 15px; /* Kurangi padding radio group */
        }
        .stRadio [data-baseweb="radio"] {
            min-width: unset; /* Hapus min-width agar lebih fleksibel */
            padding: 8px 15px; /* Perkecil padding item radio */
            font-size: 15px; /* Perkecil font item radio */
            width: 100%; /* Pastikan setiap opsi radio mengisi lebar penuh */
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
if 'data_needs_recalc' not in st.session_state: # New flag for recalculation
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

# Function to generate the Plotly graph
def create_abrasion_plot(x_values, y_values, results, analysis_choice):
    fig = go.Figure()

    # Add Abrasion Data Curve (always)
    if not x_values.empty and not y_values.empty:
        fig.add_trace(go.Scatter(
            x=x_values, 
            y=y_values,
            mode='lines+markers',
            name='Data Abrasi',
            line=dict(color='#8B4513', width=3),
            marker=dict(size=8, color='#DAA520')
        ))

        # Add Vertical Line at x=50 (always)
        plot_y_min = y_values.min() if not y_values.empty else 0
        plot_y_max = y_values.max() if not y_values.empty else 1000 # Fallback for empty data
        y_range_span = plot_y_max - plot_y_min
        y0_line = plot_y_min - y_range_span * 0.1 if y_range_span > 0 else 0
        y1_line = plot_y_max + y_range_span * 0.1 if y_range_span > 0 else 1000

        fig.add_shape(
            type="line",
            x0=TARGET_X_VALUE, y0=y0_line,
            x1=TARGET_X_VALUE, y1=y1_line,
            line=dict(color="#FF4500", width=2, dash="dash"),
            layer="below" # Ensure line is behind data points
        )
        fig.add_annotation(
            x=TARGET_X_VALUE, y=y1_line * 0.95,
            text=f"x={TARGET_X_VALUE}", showarrow=False,
            font=dict(color="#FF4500", size=14, family="Montserrat, sans-serif", weight="bold"),
            bgcolor="rgba(26,26,26,0.7)", bordercolor="#FF4500", borderwidth=1, borderpad=4
        )

        # Add specific lines based on exact choice
        if analysis_choice == "Garis Titik 10 & 20":
            if results.get('pt10_20_line_x_range', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['pt10_20_line_x_range'],
                    y=results['pt10_20_line_y'],
                    mode='lines',
                    name='Garis Titik 10 & 20',
                    line=dict(color='#ADD8E6', width=2, dash='dot') # Light blue dotted
                ))
                # Add points for 10th and 20th data point if they exist
                if not np.isnan(results['specific_x1_pt10_20']):
                    fig.add_trace(go.Scatter(
                        x=[results['specific_x1_pt10_20']],
                        y=[results['specific_y1_pt10_20']],
                        mode='markers',
                        name='Titik ke-10',
                        marker=dict(size=10, color='#ADD8E6', symbol='circle')
                    ))
                if not np.isnan(results['specific_x2_pt10_20']):
                    fig.add_trace(go.Scatter(
                        x=[results['specific_x2_pt10_20']],
                        y=[results['specific_y2_pt10_20']],
                        mode='markers',
                        name='Titik ke-20',
                        marker=dict(size=10, color='#ADD8E6', symbol='circle')
                    ))
            if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE], y=[results['y_at_x_50_pt10_20_line']],
                    mode='markers',
                    name='Potongan Garis 10-20 di x=50',
                    marker=dict(size=12, color='#ADD8E6', symbol='star'),
                    hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"
                ))
        
        elif analysis_choice == "Garis yang melewati banyak titik":
            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['ransac_line_x'],
                    y=results['ransac_line_y'],
                    mode='lines',
                    name='Regresi RANSAC',
                    line=dict(color='#90EE90', width=2, dash='dash') # Light green dashed
                ))
            if not np.isnan(results.get('y_at_x_50_ransac_line')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE], y=[results['y_at_x_50_ransac_line']],
                    mode='markers',
                    name='Potongan RANSAC di x=50',
                    marker=dict(size=12, color='#90EE90', symbol='star'),
                    hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"
                ))

        elif analysis_choice == "Kurva Data Asli": # New choice for original curve
            if not np.isnan(results.get('y_at_x_50_original_curve')):
                 fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE], y=[results['y_at_x_50_original_curve']],
                    mode='markers',
                    name='Potongan Kurva Asli di x=50',
                    marker=dict(size=12, color='#DAA520', symbol='star'),
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
                    line=dict(color='#ADD8E6', width=2, dash='dot')
                ))
                if not np.isnan(results['specific_x1_pt10_20']):
                    fig.add_trace(go.Scatter(x=[results['specific_x1_pt10_20']], y=[results['specific_y1_pt10_20']], mode='markers', name='Titik ke-10', marker=dict(size=10, color='#ADD8E6', symbol='circle')))
                if not np.isnan(results['specific_x2_pt10_20']):
                    fig.add_trace(go.Scatter(x=[results['specific_x2_pt10_20']], y=[results['specific_y2_pt10_20']], mode='markers', name='Titik ke-20', marker=dict(size=10, color='#ADD8E6', symbol='circle')))
                if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
                    fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_pt10_20_line']], mode='markers', name='Potongan Garis 10-20 di x=50', marker=dict(size=12, color='#ADD8E6', symbol='star'), hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))

            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['ransac_line_x'],
                    y=results['ransac_line_y'],
                    mode='lines',
                    name='Regresi RANSAC',
                    line=dict(color='#90EE90', width=2, dash='dash')
                ))
            if not np.isnan(results.get('y_at_x_50_ransac_line')):
                fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_ransac_line']], mode='markers', name='Potongan RANSAC di x=50', marker=dict(size=12, color='#90EE90', symbol='star'), hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))
            
            if not np.isnan(results.get('y_at_x_50_original_curve')):
                 fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_original_curve']], mode='markers', name='Potongan Kurva Asli di x=50', marker=dict(size=12, color='#DAA520', symbol='star'), hovertemplate=f"<b>Potongan (Kurva Asli)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))


    # Update layout for dark mode
    fig.update_layout(
        title={
            'text': 'Grafik Abrasi Benang',
            'yref': 'paper', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(color='#F8F8F8', size=24, family='Playfair Display, serif')
        },
        xaxis_title='Nilai X',
        yaxis_title='Nilai Benang Putus (N)',
        plot_bgcolor='#1A1A1A', # Background plot
        paper_bgcolor='#1A1A1A', # Background di luar plot
        font=dict(color='#E0E0E0', family='Montserrat, sans-serif'), # FIX: combined font family
        xaxis=dict(
            showgrid=True, gridcolor='#282828', zeroline=False,
            title_font=dict(size=18), tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#282828', zeroline=False,
            title_font=dict(size=18), tickfont=dict(size=14)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(26,26,26,0.7)", bordercolor="#282828", borderwidth=1,
            font=dict(size=14)
        ),
        hovermode="x unified", # Better hover experience
        margin=dict(l=40, r=40, b=40, t=100) # Adjust margins for title
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
        hide_index=False, # Tampilkan indeks
        column_config={
            "x_value": st.column_config.NumberColumn("Nilai Tetap (x)", format="%.1f", help="Nilai X ini adalah titik pengukuran standar dan tidak dapat diubah."),
            "y_value": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f", help="Nilai benang putus atau gaya putus dalam Newton (N)"),
        },
        num_rows="dynamic", # Allow adding/deleting rows
        use_container_width=True,
        key="data_editor",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes", use_container_width=True):
            try:
                # Get data, handle potential empty rows from dynamic editor
                # Filter out rows where both x and y are NaN if dynamic rows are added
                cleaned_edited_df = edited_df.dropna(subset=['x_value', 'y_value'])

                if not np.all(np.diff(cleaned_edited_df['x_value'].values) > 0):
                    st.error("Nilai 'x_value' harus monoton meningkat. Harap perbaiki data Anda.")
                elif cleaned_edited_df.empty:
                    st.warning("Tabel data kosong. Harap masukkan data.")
                else:
                    # Make sure x_values from original_data are preserved if rows are added/deleted
                    # We only allow editing of y_values, x_values are fixed based on INITIAL_DATA
                    # If rows were added, they will have their x_values set to default 0.0, we need to handle that.
                    # A more robust approach would be to enforce the x_values count or disable row additions for fixed x_values.
                    
                    if len(cleaned_edited_df) != len(INITIAL_DATA['x_values']):
                        st.warning("Jumlah baris data telah berubah. Pastikan Anda hanya mengubah 'Nilai Benang Putus (N)' pada data yang sudah ada atau impor data dengan struktur yang sesuai.")
                        # Reset to initial data or handle this case specifically
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
                # Ensure data types are numeric
                df_uploaded['x_values'] = pd.to_numeric(df_uploaded['x_values'], errors='coerce')
                df_uploaded['y_values'] = pd.to_numeric(df_uploaded['y_values'], errors='coerce')
                
                # Drop rows with NaN in critical columns after conversion
                df_uploaded.dropna(subset=['x_values', 'y_values'], inplace=True)

                if not np.all(np.diff(df_uploaded['x_values'].values) > 0):
                    st.error("Nilai 'x_values' dari file Excel harus monoton meningkat. Harap perbaiki data Anda.")
                elif df_uploaded.empty:
                    st.warning("File Excel kosong atau tidak mengandung data yang valid setelah pembersihan.")
                else:
                    st.session_state.data = df_uploaded[['x_values', 'y_values']]
                    st.session_state.data_needs_recalc = True
                    st.success("Data dari Excel berhasil diimpor!")
                    st.dataframe(st.session_state.data.head(), use_container_width=True) # Show preview
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
    st.session_state.data_needs_recalc = False # Reset flag

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

# Tampilkan hanya satu kartu hasil berdasarkan pilihan analysis_choice
if analysis_choice == "Kurva Data Asli":
    st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Kurva Data Asli</h3>", unsafe_allow_html=True)
    if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #DAA520;'>{st.session_state.calculated_results['y_at_x_50_original_curve']:.2f} N</p>", unsafe_allow_html=True)
        st.markdown("<p><i>Interpolasi linear dari kurva data asli pada X=50.</i></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #B0B0B0;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif analysis_choice == "Garis Titik 10 & 20":
    st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Garis Titik 10 & 20</h3>", unsafe_allow_html=True)
    if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #ADD8E6;'>{st.session_state.calculated_results['y_at_x_50_pt10_20_line']:.2f} N</p>", unsafe_allow_html=True)
        st.markdown("<p><i>Regresi linear yang melewati titik ke-10 dan ke-20 pada X=50.</i></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #B0B0B0;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif analysis_choice == "Garis yang melewati banyak titik":
    st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Garis RANSAC</h3>", unsafe_allow_html=True)
    if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
        st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #90EE90;'>{st.session_state.calculated_results['y_at_x_50_ransac_line']:.2f} N</p>", unsafe_allow_html=True)
        st.markdown("<p><i>Regresi robust RANSAC pada X=50, cocok untuk data dengan outlier.</i></p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #B0B0B0;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif analysis_choice == "Tampilkan Semua":
    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        with st.container(height=180):
            st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Kurva Data Asli</h3>", unsafe_allow_html=True)
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
                st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #DAA520;'>{st.session_state.calculated_results['y_at_x_50_original_curve']:.2f} N</p>", unsafe_allow_html=True)
                st.markdown("<p><i>Interpolasi linear dari kurva data asli pada X=50.</i></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #B0B0B0;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_res2:
        with st.container(height=180):
            st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Garis Titik 10 & 20</h3>", unsafe_allow_html=True)
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
                st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #ADD8E6;'>{st.session_state.calculated_results['y_at_x_50_pt10_20_line']:.2f} N</p>", unsafe_allow_html=True)
                st.markdown("<p><i>Regresi linear yang melewati titik ke-10 dan ke-20.</i></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #B0B0B0;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_res3:
        with st.container(height=180):
            st.markdown("<div class='dark-card result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Garis RANSAC</h3>", unsafe_allow_html=True)
            if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
                st.markdown(f"<p style='font-size: 32px; font-weight: bold; color: #90EE90;'>{st.session_state.calculated_results['y_at_x_50_ransac_line']:.2f} N</p>", unsafe_allow_html=True)
                st.markdown("<p><i>Regresi robust terhadap semua titik data.</i></p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #B0B0B0;'>Tidak dapat dihitung</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang - Dibuat oleh Radix Indonesia
</div>
""", unsafe_allow_html=True)
