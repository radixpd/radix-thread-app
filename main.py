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
    layout="wide" # Tetap wide untuk desktop, akan dihandle media query untuk mobile
)

# --- CSS Kustom untuk Tampilan Dark Mode Minimalis & Elegan (Revisi Tambahan untuk Responsif) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap');

    /* Target elemen HTML dan body untuk memastikan background hitam total */
    html, body {
        background-color: #0A0A0A !important;
        color: #E0E0E0; /* Pastikan teks juga terang */
        overflow-x: hidden; /* Mencegah scrolling horizontal yang tidak diinginkan */
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
        font-size: 16px; /* Ukuran desktop */
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
            font-size: 14px; /* Perkecil ukuran font teks biasa */
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
            font-size: 13px;
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
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        st.markdown("<h2 style='text-align: center;'>Akses Aplikasi</h2>", unsafe_allow_html=True)
        password_input = st.text_input("Masukkan Kode Akses Anda", type="password", key="password_input", help="Hubungi administrator untuk kode akses.")
        
        # Kolom untuk tombol "Masuk" agar berada di tengah dan responsif
        col_pw1, col_pw2, col_pw3 = st.columns([1,1,1])
        with col_pw2:
            # Gunakan st.session_state.password_entered = True langsung di dalam if
            if st.button("Masuk", key="login_button", use_container_width=True):
                if password_input == ACCESS_CODE:
                    st.session_state.password_entered = True
                    st.rerun() # Refresh halaman untuk menampilkan aplikasi
                else:
                    st.error("Kode akses salah. Silakan coba lagi.")
        st.markdown("<br><br>", unsafe_allow_html=True)
        return False
    return True

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

# --- Data Awal ---
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# Inisialisasi session state
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)
if 'update_graph' not in st.session_state:
    st.session_state.update_graph = False
if 'calculated_results' not in st.session_state:
    st.session_state.calculated_results = {}

# --- Fungsi untuk Menghitung Garis dan Titik (dengan Cache) ---
@st.cache_data(show_spinner="Menghitung analisis data...")
def calculate_lines_and_points(x_values, y_values):
    results = {
        'y_at_x_50_original_curve': np.nan,
        'specific_x1_pt10_20': np.nan, 'specific_y1_pt10_20': np.nan,
        'specific_x2_pt10_20': np.nan, 'specific_y2_pt10_20': np.nan,
        'y_at_x_50_pt10_20_line': np.nan,
        'pt10_20_line_x_range': np.array([]), 'pt10_20_line_y': np.array([]),
        'y_at_x_50_ransac_line': np.nan,
        'ransac_line_x': np.array([]), 'ransac_line_y': np.array([])
    }

    if len(x_values) < 2 or len(y_values) < 2:
        return results

    # Original curve interpolation
    f = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    results['y_at_x_50_original_curve'] = float(f(50))

    # Garis Antara Titik 10 & 20
    if len(x_values) >= 20:
        results['specific_x1_pt10_20'] = x_values.iloc[9]
        results['specific_y1_pt10_20'] = y_values.iloc[9]
        results['specific_x2_pt10_20'] = x_values.iloc[19]
        results['specific_y2_pt10_20'] = y_values.iloc[19]
    elif len(x_values) >= 2: # Fallback jika data kurang dari 20 
        results['specific_x1_pt10_20'] = x_values.iloc[0]
        results['specific_y1_pt10_20'] = y_values.iloc[0]
        results['specific_x2_pt10_20'] = x_values.iloc[-1]
        results['specific_y2_pt10_20'] = y_values.iloc[-1]
        st.info("Dataset kurang dari 20 titik. Garis 'Titik 10 & 20' dihitung antara titik pertama dan terakhir.")
    
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']) and results['specific_x1_pt10_20'] != results['specific_x2_pt10_20']:
        slope_pt10_20 = (results['specific_y2_pt10_20'] - results['specific_y1_pt10_20']) / (results['specific_x2_pt10_20'] - results['specific_x1_pt10_20'])
        intercept_pt10_20 = results['specific_y1_pt10_20'] - slope_pt10_20 * results['specific_x1_pt10_20']
        results['y_at_x_50_pt10_20_line'] = slope_pt10_20 * 50 + intercept_pt10_20

        x_min_plot = x_values.min() if not x_values.empty else 0
        x_max_plot = x_values.max() if not x_values.empty else 100
        results['pt10_20_line_x_range'] = np.linspace(min(x_min_plot, 50), max(x_max_plot, 50), 100)
        results['pt10_20_line_y'] = slope_pt10_20 * results['pt10_20_line_x_range'] + intercept_pt10_20

    # Regresi Linear Robust (RANSAC)
    if len(x_values) >= 2:
        try:
            X_reshaped = x_values.values.reshape(-1, 1)
            y_reshaped = y_values.values
            
            residual_threshold_val = np.std(y_reshaped) * 0.5 if len(y_reshaped) > 1 and np.std(y_reshaped) > 0 else 1.0

            ransac = RANSACRegressor(LinearRegression(),
                                     min_samples=2,
                                     residual_threshold=residual_threshold_val,
                                     random_state=42,
                                     max_trials=1000)
            ransac.fit(X_reshaped, y_reshaped)
            results['y_at_x_50_ransac_line'] = ransac.predict(np.array([[50]]))[0]

            x_min_plot = x_values.min() if not x_values.empty else 0
            x_max_plot = x_values.max() if not x_values.empty else 100
            
            results['ransac_line_x'] = np.linspace(min(x_min_plot, 50), max(x_max_plot, 50), 100)
            results['ransac_line_y'] = ransac.predict(results['ransac_line_x'].reshape(-1, 1))
            
        except Exception as e:
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])

    return results

# --- Bagian Input Data ---
st.subheader("Input Data")
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.write("Ubah nilai Y (N atau nilai benang putus) dari data abrasi:")
    
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
            "x_value": st.column_config.NumberColumn("Nilai (%)", format="%.1f"),
            "y_value": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f"),
        },
        use_container_width=True,
        key="data_editor",
    )
    
    # Gunakan kolom untuk tombol agar responsif
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes", use_container_width=True):
            try:
                # Ambil data tanpa index yang sudah dimodifikasi
                st.session_state.data['y_values'] = edited_df['y_value'].astype(float).tolist()
                st.session_state.update_graph = True 
                calculate_lines_and_points.clear() # Clear cache
                st.success("Data berhasil diperbarui!")
            except ValueError:
                st.error("Pastikan semua nilai Y adalah angka yang valid.")
    
    with col2:
        if st.button("Reset Data Awal", key="reset_values", use_container_width=True):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.update_graph = True 
            calculate_lines_and_points.clear() # Clear cache
            st.success("Data berhasil direset ke nilai awal!")

with tabs[1]:
    st.write("Unggah file Excel dengan kolom 'x_values' dan 'y_values'.")
    uploaded_file = st.file_uploader("Pilih File Excel", type=['xlsx', 'xls'])
    
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
                        "x_values": st.column_config.NumberColumn("Nilai (%)", format="%.1f"),
                        "y_values": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f"),
                    }
                )
                
                if st.button("Gunakan Data Ini", key="use_imported", use_container_width=True):
                    st.session_state.data['x_values'] = df_uploaded['x_values'].astype(float).dropna()
                    st.session_state.data['y_values'] = df_uploaded['y_values'].astype(float).dropna()
                    if len(st.session_state.data['x_values']) != len(st.session_state.data['y_values']):
                        st.error("Jumlah nilai X dan Y harus sama. Periksa file Excel Anda.")
                    elif st.session_state.data['x_values'].empty:
                        st.warning("File Excel tidak memiliki data yang valid. Periksa formatnya.")
                    else:
                        st.session_state.update_graph = True
                        calculate_lines_and_points.clear() # Clear cache
                        st.success("Data impor berhasil diterapkan!")
            else:
                st.error("File Excel harus mengandung kolom 'x_values' dan 'y_values'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file Excel: {e}")

st.markdown("---") # Garis pemisah

# Download Sample Excel Template
if st.button("Unduh Template Excel", use_container_width=True):
    sample_df = pd.DataFrame(INITIAL_DATA)
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Klik untuk Mengunduh",
        data=buffer,
        file_name="template_abrasi_benang.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---") # Garis pemisah

# --- Bagian Grafik dan Hasil ---

# Perbarui perhitungan jika data berubah
if st.session_state.update_graph or not st.session_state.calculated_results:
    x_values = pd.Series(st.session_state.data['x_values'])
    y_values = pd.Series(st.session_state.data['y_values'])
    
    if len(x_values) < 2 or len(y_values) < 2:
        st.warning("Data tidak cukup untuk analisis. Masukkan minimal 2 pasangan X dan Y.")
        # Clear results if data is insufficient to avoid displaying stale results
        st.session_state.calculated_results = {}
        st.stop()
    
    st.session_state.calculated_results = calculate_lines_and_points(x_values, y_values)
    st.session_state.update_graph = False

results = st.session_state.calculated_results
x_values = pd.Series(st.session_state.data['x_values'])
y_values = pd.Series(st.session_state.data['y_values'])


st.subheader("Visualisasi & Hasil Analisis")

# UNIFIED RADIO BUTTON UNTUK GRAFIK DAN HASIL 
analysis_choice = st.radio(
    "Pilih jenis analisis yang ingin ditampilkan:",
    ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis yang paling banyak melewati titik", "Tampilkan Semua"),
    key="analysis_choice",
    horizontal=True # Biarkan horizontal untuk desktop, akan dihandle media query untuk mobile
)

# Render Grafik 
st.markdown("---") 
st.write("#### Grafik Abrasi Benang") 

fig = go.Figure()

# Tambahkan Kurva Data Abrasi (selalu) 
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_values,
    mode='lines+markers',
    name='Data Abrasi',
    line=dict(color='#8B4513', width=3), # Warna emas/tembaga untuk kurva asli
    marker=dict(size=8, color='#DAA520') # Emas gelap untuk marker
))

# Tambahkan Garis Vertikal di x=50 (selalu) 
fig.add_shape(
    type="line",
    x0=50, y0=y_values.min() * 0.9 if y_values.min() < 0 else 0,
    x1=50, y1=y_values.max() * 1.1,
    line=dict(color="#FF4500", width=2, dash="dash"), # Oranye kemerahan yang kuat
)
# Adjusted annotation position for better visibility if lines overlap 
fig.add_annotation(
    x=50, y=y_values.max() * 1.05, text="x=50", showarrow=False,
    font=dict(color="#FF4500", size=14, family="Montserrat, sans-serif", weight="bold")
)

# Tambahkan titik perpotongan kurva asli dengan x=50 (selalu) 
if not np.isnan(results['y_at_x_50_original_curve']):
    fig.add_trace(go.Scatter(
        x=[50], y=[results['y_at_x_50_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=50, y={results["y_at_x_50_original_curve"]:.2f}',
        marker=dict(size=14, color='#FF4500', symbol='circle', line=dict(width=2, color='white'))
    ))

# Kondisional untuk Garis Titik 10 & 20 
if analysis_choice in ["Garis Titik 10 & 20", "Tampilkan Semua"]:
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']):
        fig.add_trace(go.Scatter(
            x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
            y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
            mode='markers', name='Titik Referensi (10 & 20)',
            marker=dict(size=12, color='#FFD700', symbol='star', line=dict(width=2, color='white')) # Emas terang
        ))
        fig.add_trace(go.Scatter(
            x=results['pt10_20_line_x_range'], y=results['pt10_20_line_y'],
            mode='lines', name='Garis Titik 10 & 20',
            line=dict(color="#B8860B", width=3, dash="dot"), showlegend=True # Emas gelap
        ))
        if not np.isnan(results['y_at_x_50_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[50], y=[results['y_at_x_50_pt10_20_line']],
                mode='markers', name=f'Int. Garis 10-20 di x=50, y={results["y_at_x_50_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#B8860B', symbol='circle-open', line=dict(width=3, color='#B8860B'))
            ))
            y_pos_pt10_20_label = results['y_at_x_50_pt10_20_line'] + (y_values.max() * 0.05 if y_values.max() > 0 else 50)
            fig.add_annotation(
                x=50, y=y_pos_pt10_20_label, text=f"Garis 10-20: {results['y_at_x_50_pt10_20_line']:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#B8860B',
                font=dict(size=14, color='#B8860B', family="Montserrat, sans-serif"),
                bordercolor="#B8860B", borderwidth=1, borderpad=4, bgcolor="rgba(26,26,26,0.7)", opacity=0.9
            )

# Kondisional untuk Garis Regresi RANSAC 
if analysis_choice in ["Garis yang paling banyak melewati titik", "Tampilkan Semua"]:
    if not np.isnan(results['y_at_x_50_ransac_line']) and len(results['ransac_line_x']) > 0:
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'], y=results['ransac_line_y'],
            mode='lines', name='Garis yang paling banyak melewati titik',
            line=dict(color='#00CED1', width=3, dash='dash'), showlegend=True # Biru-Cyan yang elegan (mirip teal)
        ))
        fig.add_trace(go.Scatter(
            x=[50], y=[results['y_at_x_50_ransac_line']],
            mode='markers', name=f'Int. Garis yang paling banyak melewati titik di x=50, y={results["y_at_x_50_ransac_line"]:.2f}',
            marker=dict(size=14, color='#00CED1', symbol='diamond-open', line=dict(width=3, color='#00CED1'))
        ))
        y_pos_ransac_label = results['y_at_x_50_ransac_line'] - (y_values.max() * 0.05 if results['y_at_x_50_ransac_line'] > 0 else 50)
        fig.add_annotation(
            x=50, y=y_pos_ransac_label, text=f"Garis terbanyak: {results['y_at_x_50_ransac_line']:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#00CED1',
            font=dict(size=14, color='#00CED1', family="Montserrat, sans-serif"),
            bordercolor="#00CED1", borderwidth=1, borderpad=4, bgcolor="rgba(26,26,26,0.7)", opacity=0.9
        )

# Tata Letak Plotly 
fig.update_layout(
    title_text='Grafik Data Abrasi Benang vs. Jumlah Siklus',
    xaxis_title='Jumlah Siklus (%)',
    yaxis_title='Nilai Benang Putus (N)',
    hovermode="x unified",
    template="plotly_dark", # Gunakan template dark mode
    height=600,
    xaxis=dict(showgrid=True, gridcolor='#282828', zeroline=True, zerolinecolor='#282828'),
    yaxis=dict(showgrid=True, gridcolor='#282828', zeroline=True, zerolinecolor='#282828'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(26,26,26,0.7)",
        bordercolor="#282828",
        borderwidth=1,
        font=dict(color='#E0E0E0')
    ),
    plot_bgcolor='#0A0A0A', # Warna background plot
    paper_bgcolor='#0A0A0A' # Warna background kertas
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---") # Garis pemisah

# Tampilkan Hasil Perhitungan 
st.write("#### Hasil Perhitungan Titik Perpotongan (pada x=50)")

if results:
    st.markdown(f"""
    <div class="dark-card">
        <h3>Perpotongan Kurva Data Asli:</h3>
        <p>Dengan interpolasi linear, nilai Y saat X=50 adalah: <strong>{results['y_at_x_50_original_curve']:.2f} N</strong></p>
    </div>
    """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <div class="dark-card">
            <h3>Perpotongan Garis Titik 10 & 20:</h3>
            <p>Garis ini dihitung antara titik ke-10 (X={results['specific_x1_pt10_20']:.1f}, Y={results['specific_y1_pt10_20']:.2f}) dan titik ke-20 (X={results['specific_x2_pt10_20']:.1f}, Y={results['specific_y2_pt10_20']:.2f}).</p>
            <p>Nilai Y saat X=50 pada garis ini adalah: <strong>{results['y_at_x_50_pt10_20_line']:.2f} N</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="dark-card">
            <h3>Perpotongan Garis Titik 10 & 20:</h3>
            <p>Tidak dapat menghitung garis ini karena data tidak mencukupi atau titik awal/akhir sama.</p>
        </div>
        """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <div class="dark-card">
            <h3>Perpotongan Garis yang paling banyak melewati titik:</h3>
            <p>Menggunakan regresi RANSAC, sebuah metode robust yang dapat mengabaikan outlier, nilai Y saat X=50 pada garis ini adalah: <strong>{results['y_at_x_50_ransac_line']:.2f} N</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="dark-card">
            <h3>Perpotongan Garis yang paling banyak melewati titik:</h3>
            <p>Tidak dapat menghitung garis regresi RANSAC.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Tidak ada hasil yang ditampilkan. Harap masukkan data yang valid terlebih dahulu.")

# --- Footer ---
st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang oleh RADIX<br>
    &copy; 2025 Semua Hak Dilindungi.
</div>
""", unsafe_allow_html=True)
