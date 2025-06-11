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

# --- CSS Kustom untuk Tampilan Dark Mode Minimalis & Elegan (Revisi) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght=400;700&display=swap');

    /* General Styles */
    .main {
        background-color: #0A0A0A; /* Lebih gelap dari #121212 */
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif; /* Font umum yang lebih modern */
    }
    .stApp {
        max-width: 1300px; /* Sedikit lebih lebar */
        margin: 0 auto;
        padding-top: 30px; /* Padding atas lebih besar */
        padding-bottom: 50px; /* Padding bawah untuk footer */
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #F8F8F8; /* Sedikit lebih putih dari #FFFFFF */
        font-family: 'Playfair Display', serif; /* Font serif untuk judul, kesan mewah */
        letter-spacing: 0.8px; /* Jarak huruf lebih lebar */
    }
    h1 {
        font-weight: 700;
        font-size: 44px; /* Lebih besar */
        padding-bottom: 15px; /* Lebih tebal */
        border-bottom: 3px solid #8B4513; /* Warna aksen emas/tembaga gelap */
        text-align: center;
        text-shadow: 0 4px 10px rgba(0,0,0,0.4); /* Bayangan teks lebih halus */
    }
    h2 {
        font-weight: 600;
        font-size: 32px; /* Lebih besar */
        color: #DAA520; /* Emas gelap untuk subheader utama */
        margin-bottom: 20px;
        border-bottom: 1px solid #282828; /* Border lebih gelap */
        padding-bottom: 8px;
        font-family: 'Montserrat', sans-serif; /* Kembali ke sans-serif untuk subheader */
        letter-spacing: 0.5px;
    }
    h3 {
        font-weight: 600;
        font-size: 24px;
        color: #F8F8F8;
        font-family: 'Montserrat', sans-serif;
        margin-top: 25px; /* Margin atas untuk memisahkan konten */
        margin-bottom: 15px;
    }
    p, li, span {
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif;
        line-height: 1.8; /* Jarak baris lebih lega */
        font-size: 16px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #8B4513; /* Aksen emas/tembaga */
        color: white;
        border-radius: 10px; /* Lebih bulat */
        border: none;
        font-weight: 600; /* Lebih tebal */
        transition: all 0.3s ease;
        padding: 12px 25px; /* Lebih besar */
        font-size: 17px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); /* Bayangan lebih kuat */
    }
    .stButton>button:hover {
        background-color: #A0522D; /* Aksen emas/tembaga lebih terang saat hover */
        box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4); /* Bayangan aksen */
        transform: translateY(-3px); /* Efek angkat lebih jelas */
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Montserrat', sans-serif;
        color: #B0B0B0; /* Sedikit lebih gelap */
        font-weight: 600; /* Lebih tebal */
        padding: 12px 20px; /* Lebih besar */
        font-size: 17px;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 12px; /* Lebih bulat */
        background-color: #1A1A1A; /* Lebih gelap dari #1E1E1E */
        box-shadow: 0 6px 18px rgba(0,0,0,0.3); /* Bayangan lebih kuat */
        margin-bottom: 25px;
        border: 1px solid #282828; /* Border halus */
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 30px; /* Lebih besar */
        border-radius: 12px;
        background-color: #1A1A1A;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #DAA520; /* Emas gelap untuk highlight */
        border-radius: 6px;
        height: 4px; /* Garis highlight lebih tebal */
    }

    /* Radio Buttons - Unified for Graph & Results */
    .stRadio > label {
        color: #F8F8F8; /* Lebih terang */
        font-size: 18px; /* Lebih besar */
        font-weight: 600; /* Lebih tebal */
        margin-bottom: 15px;
    }
    .stRadio > div { /* Container for radio buttons */
        background-color: #1A1A1A; /* Lebih gelap */
        border-radius: 12px;
        padding: 20px; /* Lebih besar */
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        display: flex;
        flex-wrap: wrap;
        gap: 15px; /* Jarak antar item lebih besar */
    }
    .stRadio [data-baseweb="radio"] { /* Individual radio item */
        background-color: #282828; /* Lebih gelap */
        border-radius: 10px; /* Lebih bulat */
        padding: 10px 20px;
        transition: background-color 0.3s ease, border 0.3s ease;
        flex-grow: 1;
        text-align: center;
        min-width: 150px; /* Minimal lebar untuk setiap opsi */
    }
    .stRadio [data-baseweb="radio"]:hover {
        background-color: #3A3A3A;
        border: 1px solid #DAA520; /* Border aksen saat hover */
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: #DAA520 !important; /* Emas gelap saat terpilih */
        color: white;
        border: 1px solid #DAA520;
        box-shadow: 0 4px 15px rgba(218, 165, 32, 0.4); /* Bayangan aksen */
    }
    .stRadio [data-baseweb="radio"] span:last-child { /* text of the radio button */
        color: #E0E0E0;
        font-weight: 600;
        font-size: 17px;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: white;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1A1A1A; /* Lebih gelap */
        border-radius: 15px; /* Lebih bulat */
        padding: 30px; /* Lebih besar */
        box-shadow: 0 6px 25px rgba(0,0,0,0.3); /* Bayangan lebih kuat */
        margin-bottom: 30px;
        border: 1px solid #282828; /* Border halus */
    }

    /* Radix Header (PULCRA Branding) */
    .app-header {
        background: linear-gradient(145deg, #1A1A1A, #0A0A0A); /* Gradasi background */
        padding: 40px; /* Lebih besar */
        border-radius: 20px; /* Lebih bulat */
        box-shadow: 0 8px 30px rgba(0,0,0,0.5); /* Bayangan lebih kuat */
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(8px); /* Blur lebih kuat */
        border: 1px solid #282828;
    }
    .pulcra-logo {
        font-family: 'Playfair Display', serif; /* Font serif untuk logo */
        font-weight: 700;
        font-size: 56px; /* Lebih besar */
        color: #DAA520; /* Emas gelap untuk logo */
        margin-bottom: 10px;
        letter-spacing: 5px; /* Jarak huruf lebih lebar */
        text-shadow: 0 5px 20px rgba(218, 165, 32, 0.5); /* Bayangan emas yang kuat */
        text-transform: uppercase; /* Uppercase untuk logo */
    }
    .app-header h1 {
        font-size: 38px; /* Ukuran h1 di header */
        border-bottom: none; /* Hilangkan border bottom di h1 header */
        padding-bottom: 0;
        margin-bottom: 0;
        text-shadow: none; /* Hilangkan text shadow di h1 header */
        color: #F8F8F8;
    }
    .app-header p {
        font-size: 18px;
        color: #B0B0B0;
        margin-top: 10px;
        letter-spacing: 0.5px;
    }

    /* Footer */
    .radix-footer {
        text-align: center;
        margin-top: 60px; /* Margin atas lebih besar */
        padding: 25px; /* Padding lebih besar */
        font-size: 15px;
        font-family: 'Montserrat', sans-serif;
        color: #A0A0A0;
        border-top: 1px solid #282828;
        background-color: #1A1A1A; /* Senada dengan card/tabs */
        border-radius: 0 0 15px 15px; /* Lebih bulat */
        box-shadow: 0 -4px 15px rgba(0,0,0,0.3); /* Bayangan ke atas */
    }

    /* Other elements */
    hr {
        border-color: #282828 !important; /* Lebih gelap */
        margin: 40px 0 !important; /* Margin lebih besar */
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
        border-radius: 10px; /* Lebih bulat */
        overflow: hidden;
        border: 1px solid #282828;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); /* Bayangan */
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #DAA520; /* Border aksen emas */
        border-radius: 12px;
        padding: 25px; /* Lebih besar */
        background-color: #1A1A1A;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploaderDropzone"] p {
        color: #B0B0B0;
        font-size: 17px;
    }

    /* For data preview tables */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.3); /* Bayangan lebih kuat */
        background-color: #1A1A1A;
        max-height: 350px; /* Sedikit lebih tinggi */
        overflow-y: auto;
        border: 1px solid #282828;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
    }
    .stDataFrame th {
        background-color: #282828 !important; /* Lebih gelap */
        color: #DAA520 !important; /* Emas gelap untuk header tabel */
        font-weight: 700; /* Lebih tebal */
        position: sticky;
        top: 0;
        z-index: 1;
        font-size: 16px;
    }
    .stDataFrame td {
        background-color: #1A1A1A !important;
        color: #E0E0E0 !important;
        border-bottom: 1px solid #282828 !important;
        padding: 10px 15px; /* Padding sel */
    }
    /* Scrollbar for dataframes */
    .stDataFrame::-webkit-scrollbar {
        width: 10px; /* Lebih tebal */
        height: 10px;
    }
    .stDataFrame::-webkit-scrollbar-track {
        background: #1A1A1A;
    }
    .stDataFrame::-webkit-scrollbar-thumb {
        background: #DAA520; /* Emas gelap untuk scrollbar */
        border-radius: 10px;
    }
    .stDataFrame::-webkit-scrollbar-thumb:hover {
        background: #C49F3D; /* Emas sedikit lebih terang saat hover */
    }

    /* Plotly specifics for dark elegance */
    .js-plotly-plot .plotly .modebar {
        background-color: #1A1A1A !important; /* Modebar senada dengan background */
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .js-plotly-plot .plotly .modebar-btn {
        color: #DAA520 !important; /* Ikon modebar warna emas */
    }
    .js-plotly-plot .plotly .modebar-btn:hover {
        background-color: #282828 !important; /* Background hover untuk ikon */
    }

    /* Access Code styling */
    .stTextInput>div>div>input {
        background-color: #1A1A1A;
        border: 1px solid #282828;
        border-radius: 8px;
        color: #E0E0E0;
        padding: 10px 15px;
        font-size: 18px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput>label {
        font-size: 18px;
        color: #F8F8F8;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .st-emotion-cache-16txt4s { /* Ini adalah selector untuk error message Streamlit */
        background-color: #4A0000; /* Darker red */
        color: #FFCCCC; /* Lighter red text */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #8B0000;
    }
    .st-emotion-cache-zt5ig8 { /* Ini adalah selector untuk success message Streamlit */
        background-color: #004A00; /* Darker green */
        color: #CCFFCC; /* Lighter green text */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #008B00;
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
        col_pw1, col_pw2, col_pw3 = st.columns([1,1,1])
        with col_pw2:
            if st.button("Masuk", key="login_button", use_container_width=True):
                if password_input == ACCESS_CODE:
                    st.session_state.password_entered = True
                    st.rerun()
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
        'y_at_x_16_original_curve': np.nan,
        'y_at_x_50_original_curve': np.nan,
        'y_at_x_84_original_curve': np.nan,
        'specific_x1_pt10_20': np.nan, 'specific_y1_pt10_20': np.nan,
        'specific_x2_pt10_20': np.nan, 'specific_y2_pt10_20': np.nan,
        'y_at_x_16_pt10_20_line': np.nan,
        'y_at_x_50_pt10_20_line': np.nan,
        'y_at_x_84_pt10_20_line': np.nan,
        'pt10_20_line_x_range': np.array([]), 'pt10_20_line_y': np.array([]),
        'y_at_x_16_ransac_line': np.nan,
        'y_at_x_50_ransac_line': np.nan,
        'y_at_x_84_ransac_line': np.nan,
        'ransac_line_x': np.array([]), 'ransac_line_y': np.array([]),
        'sd_result': np.nan,
        'cv_result': np.nan
    }

    if len(x_values) < 2 or len(y_values) < 2:
        return results

    # Original curve interpolation for x=16, 50, 84
    f = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    results['y_at_x_16_original_curve'] = float(f(16))
    results['y_at_x_50_original_curve'] = float(f(50))
    results['y_at_x_84_original_curve'] = float(f(84))

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
        
        results['y_at_x_16_pt10_20_line'] = slope_pt10_20 * 16 + intercept_pt10_20
        results['y_at_x_50_pt10_20_line'] = slope_pt10_20 * 50 + intercept_pt10_20
        results['y_at_x_84_pt10_20_line'] = slope_pt10_20 * 84 + intercept_pt10_20

        x_min_plot = x_values.min() if not x_values.empty else 0
        x_max_plot = x_values.max() if not x_values.empty else 100
        results['pt10_20_line_x_range'] = np.linspace(min(x_min_plot, 16, 50, 84), max(x_max_plot, 16, 50, 84), 100)
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
            
            results['y_at_x_16_ransac_line'] = ransac.predict(np.array([[16]]))[0]
            results['y_at_x_50_ransac_line'] = ransac.predict(np.array([[50]]))[0]
            results['y_at_x_84_ransac_line'] = ransac.predict(np.array([[84]]))[0]

            x_min_plot = x_values.min() if not x_values.empty else 0
            x_max_plot = x_values.max() if not x_values.empty else 100
            
            results['ransac_line_x'] = np.linspace(min(x_min_plot, 16, 50, 84), max(x_max_plot, 16, 50, 84), 100)
            results['ransac_line_y'] = ransac.predict(results['ransac_line_x'].reshape(-1, 1))
            
        except Exception as e:
            # st.warning(f"Error Regresi RANSAC: {e}. Pastikan data memiliki variasi.") # Suppress this warning if it's too frequent
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])
    
    # Calculate SD and CV using RANSAC results if available
    if not np.isnan(results['y_at_x_84_ransac_line']) and not np.isnan(results['y_at_x_16_ransac_line']):
        results['sd_result'] = (results['y_at_x_84_ransac_line'] - results['y_at_x_16_ransac_line']) / 2
        
        if not np.isnan(results['y_at_x_50_ransac_line']) and results['y_at_x_50_ransac_line'] != 0:
            results['cv_result'] = (results['sd_result'] * 100) / results['y_at_x_50_ransac_line']
    
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
            "x_value": st.column_config.NumberColumn("Nilai Tetap (x)", format="%.1f"),
            "y_value": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f"),
        },
        use_container_width=True,
        key="data_editor",
    )
    
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
                        "x_values": st.column_config.NumberColumn("Nilai Tetap (x)", format="%.1f"),
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
    ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis Regresi RANSAC", "Tampilkan Semua"),
    key="analysis_choice",
    horizontal=True
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

# Tambahkan Garis Vertikal di x=16, x=50, x=84
vertical_lines_x = [16, 50, 84]
line_colors = {16: '#FF7F50', 50: '#FF4500', 84: '#FF6347'} # Coral, OrangeRed, Tomato
line_names = {16: 'x=16', 50: 'x=50', 84: 'x=84'}

for x_val in vertical_lines_x:
    fig.add_shape(
        type="line",
        x0=x_val, y0=y_values.min() * 0.9 if y_values.min() < 0 else 0,
        x1=x_val, y1=y_values.max() * 1.1,
        line=dict(color=line_colors[x_val], width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x_val, y=y_values.max() * 1.05, text=line_names[x_val], showarrow=False,
        font=dict(color=line_colors[x_val], size=14, family="Montserrat, sans-serif", weight="bold")
    )

# Tambahkan titik perpotongan kurva asli dengan x=16, 50, 84
if not np.isnan(results['y_at_x_16_original_curve']):
    fig.add_trace(go.Scatter(
        x=[16], y=[results['y_at_x_16_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=16, y={results["y_at_x_16_original_curve"]:.2f}',
        marker=dict(size=14, color=line_colors[16], symbol='circle', line=dict(width=2, color='white'))
    ))
if not np.isnan(results['y_at_x_50_original_curve']):
    fig.add_trace(go.Scatter(
        x=[50], y=[results['y_at_x_50_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=50, y={results["y_at_x_50_original_curve"]:.2f}',
        marker=dict(size=14, color=line_colors[50], symbol='circle', line=dict(width=2, color='white'))
    ))
if not np.isnan(results['y_at_x_84_original_curve']):
    fig.add_trace(go.Scatter(
        x=[84], y=[results['y_at_x_84_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=84, y={results["y_at_x_84_original_curve"]:.2f}',
        marker=dict(size=14, color=line_colors[84], symbol='circle', line=dict(width=2, color='white'))
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
        # Add intersection points for 10-20 line
        if not np.isnan(results['y_at_x_16_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[16], y=[results['y_at_x_16_pt10_20_line']],
                mode='markers', name=f'Int. Garis 10-20 di x=16, y={results["y_at_x_16_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#B8860B', symbol='square-open', line=dict(width=3, color='#B8860B'))
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
        if not np.isnan(results['y_at_x_84_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[84], y=[results['y_at_x_84_pt10_20_line']],
                mode='markers', name=f'Int. Garis 10-20 di x=84, y={results["y_at_x_84_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#B8860B', symbol='triangle-up-open', line=dict(width=3, color='#B8860B'))
            ))

# Kondisional untuk Garis Regresi RANSAC 
if analysis_choice in ["Garis Regresi RANSAC", "Tampilkan Semua"]:
    if not np.isnan(results['y_at_x_50_ransac_line']) and len(results['ransac_line_x']) > 0:
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'], y=results['ransac_line_y'],
            mode='lines', name='Regresi RANSAC',
            line=dict(color='#00CED1', width=3, dash='dash'), showlegend=True # Biru-Cyan yang elegan (mirip teal)
        ))
        # Add intersection points for RANSAC line
        if not np.isnan(results['y_at_x_16_ransac_line']):
            fig.add_trace(go.Scatter(
                x=[16], y=[results['y_at_x_16_ransac_line']],
                mode='markers', name=f'Int. RANSAC di x=16, y={results["y_at_x_16_ransac_line"]:.2f}',
                marker=dict(size=14, color='#00CED1', symbol='diamond-open', line=dict(width=3, color='#00CED1'))
            ))
        if not np.isnan(results['y_at_x_50_ransac_line']):
            fig.add_trace(go.Scatter(
                x=[50], y=[results['y_at_x_50_ransac_line']],
                mode='markers', name=f'Int. RANSAC di x=50, y={results["y_at_x_50_ransac_line"]:.2f}',
                marker=dict(size=14, color='#00CED1', symbol='diamond-open', line=dict(width=3, color='#00CED1'))
            ))
            y_pos_ransac_label = results['y_at_x_50_ransac_line'] - (y_values.max() * 0.05 if results['y_at_x_50_ransac_line'] > 0 else 50)
            fig.add_annotation(
                x=50, y=y_pos_ransac_label, text=f"RANSAC: {results['y_at_x_50_ransac_line']:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#00CED1',
                font=dict(size=14, color='#00CED1', family="Montserrat, sans-serif"),
                bordercolor="#00CED1", borderwidth=1, borderpad=4, bgcolor="rgba(26,26,26,0.7)", opacity=0.9
            )
        if not np.isnan(results['y_at_x_84_ransac_line']):
            fig.add_trace(go.Scatter(
                x=[84], y=[results['y_at_x_84_ransac_line']],
                mode='markers', name=f'Int. RANSAC di x=84, y={results["y_at_x_84_ransac_line"]:.2f}',
                marker=dict(size=14, color='#00CED1', symbol='diamond-up', line=dict(width=3, color='#00CED1'))
            ))

# Update layout for Plotly
fig.update_layout(
    xaxis_title="Nilai Tetap (x)",
    yaxis_title="Nilai Benang Putus (N)",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=12, color="#E0E0E0"), bgcolor="rgba(26,26,26,0.7)", borderwidth=1, bordercolor="#3A3A3A"
    ),
    margin=dict(l=20, r=20, t=60, b=20), height=600, template="plotly_dark", # Margin dan tinggi lebih besar
    plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A", font=dict(color="#E0E0E0", family="Montserrat, sans-serif"),
    hoverlabel=dict(bgcolor="rgba(26,26,26,0.9)", font_size=14, font_family="Montserrat, sans-serif") # Hoverlabel lebih elegan
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#282828', zeroline=True, zerolinewidth=1.5, zerolinecolor='#282828', tickfont=dict(color="#B0B0B0"))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#282828', zeroline=True, zerolinewidth=1.5, zerolinecolor='#282828', tickfont=dict(color="#B0B0B0"))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---") # Garis pemisah untuk hasil 

# Tampilkan Hasil Analisis (Sesuai dengan analysis_choice) 
st.write("#### Hasil Nilai Perpotongan") 

st.markdown(f"""
<div class="dark-card" style="text-align: center; padding: 25px; margin-bottom: 20px;">
""", unsafe_allow_html=True)

if analysis_choice == "Kurva Data Asli":
    st.markdown(f"""
    <p style="color: #B0B0B0; font-size: 16px;">Hasil Kurva Data Asli:</p>
    <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=16</p>
            <h2 style="color: {line_colors[16]}; font-size: 38px; margin: 5px 0;">{results['y_at_x_16_original_curve']:.2f}</h2>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=50</p>
            <h2 style="color: {line_colors[50]}; font-size: 38px; margin: 5px 0;">{results['y_at_x_50_original_curve']:.2f}</h2>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=84</p>
            <h2 style="color: {line_colors[84]}; font-size: 38px; margin: 5px 0;">{results['y_at_x_84_original_curve']:.2f}</h2>
        </div>
    </div>
    <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">Ini adalah interpolasi linear langsung dari data yang Anda masukkan.</div>
    """, unsafe_allow_html=True)

elif analysis_choice == "Garis Titik 10 & 20":
    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=16</p>
                <h2 style="color: #B8860B; font-size: 38px; margin: 5px 0;">{results['y_at_x_16_pt10_20_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=50</p>
                <h2 style="color: #B8860B; font-size: 38px; margin: 5px 0;">{results['y_at_x_50_pt10_20_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=84</p>
                <h2 style="color: #B8860B; font-size: 38px; margin: 5px 0;">{results['y_at_x_84_pt10_20_line']:.2f}</h2>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">Berdasarkan garis linear yang ditarik antara titik data ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik data ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f}).</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <h1 style="color: #B8860B; font-size: 60px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">
            Tidak cukup data untuk menghitung garis ini (membutuhkan setidaknya 20 titik).
        </div>
        """, unsafe_allow_html=True)

elif analysis_choice == "Garis Regresi RANSAC":
    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=16</p>
                <h2 style="color: #00CED1; font-size: 38px; margin: 5px 0;">{results['y_at_x_16_ransac_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=50</p>
                <h2 style="color: #00CED1; font-size: 38px; margin: 5px 0;">{results['y_at_x_50_ransac_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai di x=84</p>
                <h2 style="color: #00CED1; font-size: 38px; margin: 5px 0;">{results['y_at_x_84_ransac_line']:.2f}</h2>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">Berdasarkan model Regresi Linear Robust (RANSAC), yang secara cerdas menemukan garis terbaik dengan mengabaikan data outlier untuk menghasilkan prediksi yang lebih stabil.</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h1 style="color: #00CED1; font-size: 60px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">
            Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC (membutuhkan minimal 2 titik).
        </div>
        """, unsafe_allow_html=True)

elif analysis_choice == "Tampilkan Semua":
    st.markdown(f"""
    <p style="color: #B0B0B0; font-size: 16px;">Hasil Kurva Data Asli:</p>
    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=16</p>
            <h3 style="color: {line_colors[16]}; font-size: 28px; margin: 5px 0;">{results['y_at_x_16_original_curve']:.2f}</h3>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=50</p>
            <h3 style="color: {line_colors[50]}; font-size: 28px; margin: 5px 0;">{results['y_at_x_50_original_curve']:.2f}</h3>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=84</p>
            <h3 style="color: {line_colors[84]}; font-size: 28px; margin: 5px 0;">{results['y_at_x_84_original_curve']:.2f}</h3>
        </div>
    </div>
    <hr style="border-color: #3A3A3A !important; margin: 15px 0 !important;">
    """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=16</p>
                <h3 style="color: #B8860B; font-size: 28px; margin: 5px 0;">{results['y_at_x_16_pt10_20_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=50</p>
                <h3 style="color: #B8860B; font-size: 28px; margin: 5px 0;">{results['y_at_x_50_pt10_20_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=84</p>
                <h3 style="color: #B8860B; font-size: 28px; margin: 5px 0;">{results['y_at_x_84_pt10_20_line']:.2f}</h3>
            </div>
        </div>
        <hr style="border-color: #3A3A3A !important; margin: 15px 0 !important;">
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <h3 style="color: #B8860B; font-size: 28px; margin: 10px 0;">N/A</h3>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data untuk menghitung garis ini (membutuhkan setidaknya 20 titik).</p>
        <hr style="border-color: #3A3A3A !important; margin: 15px 0 !important;">
        """, unsafe_allow_html=True)
    
    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=16</p>
                <h3 style="color: #00CED1; font-size: 28px; margin: 5px 0;">{results['y_at_x_16_ransac_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=50</p>
                <h3 style="color: #00CED1; font-size: 28px; margin: 5px 0;">{results['y_at_x_50_ransac_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">x=84</p>
                <h3 style="color: #00CED1; font-size: 28px; margin: 5px 0;">{results['y_at_x_84_ransac_line']:.2f}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h3 style="color: #00CED1; font-size: 28px; margin: 10px 0;">N/A</h3>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.</p>
        """, unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)

# --- Perhitungan SD dan CV ---
st.write("#### Perhitungan SD dan CV (Berdasarkan Garis RANSAC)")

st.markdown("""
<div class="dark-card">
""", unsafe_allow_html=True)

if not np.isnan(results['sd_result']) and not np.isnan(results['cv_result']):
    st.markdown(f"""
    <p style="font-size: 18px; color: #DAA520; font-weight: 600;">Standard Deviation (SD):</p>
    <p style="font-size: 22px; color: #E0E0E0; font-weight: 700;">
        (Hasil Perpotongan X=84 - Hasil Perpotongan X=16) / 2
    </p>
    <p style="font-size: 26px; color: #00CED1; font-weight: 700;">
        ({results['y_at_x_84_ransac_line']:.2f} - {results['y_at_x_16_ransac_line']:.2f}) / 2 = {results['sd_result']:.2f}
    </p>
    <br>
    <p style="font-size: 18px; color: #DAA520; font-weight: 600;">Coefficient of Variation (CV):</p>
    <p style="font-size: 22px; color: #E0E0E0; font-weight: 700;">
        (SD * 100) / Hasil Perpotongan X=50
    </p>
    <p style="font-size: 26px; color: #00CED1; font-weight: 700;">
        ({results['sd_result']:.2f} * 100) / {results['y_at_x_50_ransac_line']:.2f} = {results['cv_result']:.2f}%
    </p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <p style="font-size: 18px; color: #B0B0B0;">
        Perhitungan SD dan CV memerlukan hasil perpotongan X=16, X=50, dan X=84 dari garis Regresi RANSAC. <br>
        Pastikan data Anda memadai dan analisis RANSAC berhasil.
    </p>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# --- Tampilan Tabel Data Lengkap --- 
with st.expander("Lihat Tabel Data Lengkap"): 
    # Menampilkan index dari 1 untuk tabel data lengkap 
    display_df_full = pd.DataFrame({ 
        'Nilai Tetap (x)': st.session_state.data['x_values'], 
        'Nilai Benang Putus (N)': st.session_state.data['y_values'] 
    }) 
    display_df_full.index = display_df_full.index + 1 # Ubah indeks menjadi dari 1 
    st.dataframe( 
        display_df_full, 
        hide_index=False, # Tampilkan indeks 
        use_container_width=True, 
        height=400 # Atur tinggi agar bisa di-scroll jika data banyak 
    ) 

# --- Informasi & Footer --- 
st.markdown(""" 
<div class="dark-card"> 
    <h3>Tentang Grafik</h3> 
    <p>Grafik ini menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y). Anda dapat memilih jenis analisis yang ingin ditampilkan menggunakan pilihan di atas.</p> 
    <ul> 
        <li><strong style="color: #DAA520;">Kurva Data Asli:</strong> Garis emas/tembaga mewakili data abrasi benang yang Anda masukkan. Ini adalah interpolasi linear sederhana antara setiap titik data.</li> 
        <li><strong style="color: #B8860B;">Garis Titik 10 & 20:</strong> Garis putus-putus berwarna emas gelap yang dihitung berdasarkan dua titik data spesifik: titik ke-10 dan titik ke-20. Metode ini sering digunakan dalam standar industri tertentu.</li> 
        <li><strong style="color: #00CED1;">Garis Regresi RANSAC:</strong> Garis putus-putus berwarna biru-cyan ini adalah hasil dari Regresi Linear Robust (RANSAC). RANSAC dirancang untuk mengabaikan data outlier (pencilan) dan menemukan garis tren terbaik dari sebagian besar data yang "inlier", menjadikannya pilihan yang kuat untuk data yang mungkin bising atau memiliki anomali.</li> 
    </ul> 
    <p>Setiap titik perpotongan di x=16, x=50, dan x=84 ditandai pada grafik dengan warna yang sesuai dengan garisnya, dan nilainya ditampilkan di bagian "Hasil Nilai Perpotongan".</p>
    <h3>Tentang SD & CV</h3>
    <p>
        <strong style="color: #DAA520;">Standard Deviation (SD):</strong> Mengukur penyebaran data relatif terhadap nilai rata-rata. Dalam konteks ini, dihitung sebagai setengah dari rentang antara hasil perpotongan di x=84 dan x=16 dari garis RANSAC.
    </p>
    <p>
        <strong style="color: #DAA520;">Coefficient of Variation (CV):</strong> Mengukur variabilitas relatif. Dihitung sebagai (SD * 100) dibagi dengan hasil perpotongan di x=50 dari garis RANSAC. CV yang lebih rendah menunjukkan konsistensi data yang lebih tinggi.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang - Dibuat oleh PULCRA Chemicals <br>
    © 2025 Semua Hak Dilindungi.
</div>
""", unsafe_allow_html=True)
