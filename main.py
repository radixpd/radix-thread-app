import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor, LinearRegression
import io

# --- Konfigurasi Halaman (Paling Awal) ---
st.set_page_config(
    page_title="Analisis Benang Abrasi",
    page_icon="🧵",
    layout="wide"
)

# --- CSS Kustom untuk Tampilan Dark Mode Minimalis & Elegan ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght0&display=swap');

    /* General Styles */
    .main {
        background-color: #0A0A0A;
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif;
    }
    .stApp {
        max-width: 1300px;
        margin: 0 auto;
        padding-top: 30px;
        padding-bottom: 50px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #F8F8F8;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
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
        color: #DAA520;
        margin-bottom: 20px;
        border-bottom: 1px solid #282828;
        padding-bottom: 8px;
        font-family: 'Montserrat', sans-serif;
        letter-spacing: 0.5px;
    }
    h3 {
        font-weight: 600;
        font-size: 24px;
        color: #F8F8F8;
        font-family: 'Montserrat', sans-serif;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    p, li, span {
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif;
        line-height: 1.8;
        font-size: 16px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #8B4513;
        color: white;
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
        color: #B0B0B0;
        font-weight: 600;
        padding: 12px 20px;
        font-size: 17px;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 12px;
        background-color: #1A1A1A;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        border: 1px solid #282828;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 30px;
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
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stRadio > div { /* Container for radio buttons */
        background-color: #1A1A1A;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.3);
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .stRadio [data-baseweb="radio"] { /* Individual radio item */
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
        font-size: 17px;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: white;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1A1A1A;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        margin-bottom: 30px;
        border: 1px solid #282828;
    }

    /* Radix Header (PULCRA Branding) */
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
        border: 1px solid #282828;
    }
    .pulcra-logo {
        font-family: 'Playfair Display', serif;
        font-weight: 700;
        font-size: 56px;
        color: #DAA520;
        margin-bottom: 10px;
        letter-spacing: 5px;
        text-shadow: 0 5px 20px rgba(218, 165, 32, 0.5);
        text-transform: uppercase;
    }
    .app-header h1 {
        font-size: 38px;
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0;
        text-shadow: none;
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
        margin-top: 60px;
        padding: 25px;
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
        overflow: hidden;
        border: 1px solid #282828;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #DAA520;
        border-radius: 12px;
        padding: 25px;
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
        box-shadow: 0 5px 18px rgba(0,0,0,0.3);
        background-color: #1A1A1A;
        max-height: 350px;
        overflow-y: auto;
        border: 1px solid #282828;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
    }
    .stDataFrame th {
        background-color: #282828 !important;
        color: #DAA520 !important;
        font-weight: 700;
        position: sticky;
        top: 0;
        z-index: 1;
        font-size: 16px;
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
        height: 10px;
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
        font-size: 18px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput>label {
        font-size: 18px;
        color: #F8F8F8;
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
    <h1 style="margin-top: 0;">Analisis Benang Abrasi</h1>
    <p>Alat mudah untuk visualisasi data dan hitung titik penting</p>
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
@st.cache_data(show_spinner="Menghitung data Anda...")
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

    # Interpolasi Kurva Asli untuk x=16, 50, 84
    f = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    results['y_at_x_16_original_curve'] = float(f(16))
    results['y_at_x_50_original_curve'] = float(f(50))
    results['y_at_x_84_original_curve'] = float(f(84))

    # Garis Antara Titik ke-10 & ke-20
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
        st.info("Dataset kurang dari 20 titik. Garis 'Titik ke-10 & ke-20' dihitung antara titik pertama dan terakhir.")
    
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
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])
    
    # Hitung SD dan CV berdasarkan RANSAC
    if not np.isnan(results['y_at_x_84_ransac_line']) and not np.isnan(results['y_at_x_16_ransac_line']):
        results['sd_result'] = (results['y_at_x_84_ransac_line'] - results['y_at_x_16_ransac_line']) / 2
        
        if not np.isnan(results['y_at_x_50_ransac_line']) and results['y_at_x_50_ransac_line'] != 0:
            results['cv_result'] = (results['sd_result'] * 100) / results['y_at_x_50_ransac_line']
    
    return results

# --- Bagian Input Data ---
st.subheader("1. Masukkan Data")
tabs = st.tabs(["Input Langsung", "Impor dari Excel"])

with tabs[0]:
    st.write("Silakan ubah nilai 'Benang Putus (N)' sesuai data Anda:")
    
    edited_data = pd.DataFrame({
        'Siklus (x)': st.session_state.data['x_values'],
        'Benang Putus (N)': st.session_state.data['y_values']
    })
    edited_data.index = edited_data.index + 1 # Ubah indeks menjadi dari 1
    
    edited_df = st.data_editor(
        edited_data,
        disabled=["Siklus (x)"],
        hide_index=False,
        column_config={
            "Siklus (x)": st.column_config.NumberColumn("Siklus (x)", format="%.1f"),
            "Benang Putus (N)": st.column_config.NumberColumn("Benang Putus (N)", format="%.2f"),
        },
        use_container_width=True,
        key="data_editor",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simpan Perubahan", key="apply_changes", use_container_width=True):
            try:
                st.session_state.data['y_values'] = edited_df['Benang Putus (N)'].astype(float).tolist()
                st.session_state.update_graph = True 
                calculate_lines_and_points.clear() # Hapus cache
                st.success("Data berhasil disimpan!")
            except ValueError:
                st.error("Pastikan semua nilai 'Benang Putus (N)' adalah angka yang valid.")
    
    with col2:
        if st.button("Reset Data Contoh", key="reset_values", use_container_width=True):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.update_graph = True 
            calculate_lines_and_points.clear() # Hapus cache
            st.success("Data berhasil direset ke contoh awal!")

with tabs[1]:
    st.write("Unggah file Excel Anda. Pastikan ada kolom 'x_values' dan 'y_values'.")
    uploaded_file = st.file_uploader("Pilih File Excel", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            if 'x_values' in df_uploaded.columns and 'y_values' in df_uploaded.columns:
                st.write("Pratinjau Data yang Diimpor:")
                df_uploaded_display = df_uploaded.copy()
                df_uploaded_display.index = df_uploaded_display.index + 1
                st.dataframe(
                    df_uploaded_display,
                    use_container_width=True,
                    height=300,
                    column_config={
                        "x_values": st.column_config.NumberColumn("Siklus (x)", format="%.1f"),
                        "y_values": st.column_config.NumberColumn("Benang Putus (N)", format="%.2f"),
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
                        calculate_lines_and_points.clear() # Hapus cache
                        st.success("Data impor berhasil diterapkan!")
            else:
                st.error("File Excel harus mengandung kolom 'x_values' dan 'y_values'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file Excel: {e}")

st.markdown("---")

# Download Sample Excel Template
if st.button("Unduh Contoh File Excel", use_container_width=True):
    sample_df = pd.DataFrame(INITIAL_DATA)
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Klik untuk Mengunduh",
        data=buffer,
        file_name="contoh_data_abrasi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("---")

# --- Bagian Grafik dan Hasil ---

# Perbarui perhitungan jika data berubah
if st.session_state.update_graph or not st.session_state.calculated_results:
    x_values = pd.Series(st.session_state.data['x_values'])
    y_values = pd.Series(st.session_state.data['y_values'])
    
    if len(x_values) < 2 or len(y_values) < 2:
        st.warning("Data tidak cukup untuk analisis. Masukkan minimal 2 pasang data (Siklus dan Benang Putus).")
        st.session_state.calculated_results = {}
        st.stop()
    
    st.session_state.calculated_results = calculate_lines_and_points(x_values, y_values)
    st.session_state.update_graph = False

results = st.session_state.calculated_results
x_values = pd.Series(st.session_state.data['x_values'])
y_values = pd.Series(st.session_state.data['y_values'])


st.subheader("2. Lihat Visualisasi & Hasil Analisis")

# PILIH JENIS ANALISIS
analysis_choice = st.radio(
    "Pilih apa yang ingin Anda lihat di grafik dan hasil:",
    ("Kurva Asli", "Garis Titik ke-10 & ke-20", "Garis RANSAC", "Tampilkan Semua"),
    key="analysis_choice",
    horizontal=True
)

st.markdown("---")
st.write("#### Grafik Abrasi Benang")

fig = go.Figure()

# Tambahkan Kurva Data Abrasi (selalu)
fig.add_trace(go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines+markers',
    name='Data Asli',
    line=dict(color='#8B4513', width=3),
    marker=dict(size=8, color='#DAA520')
))

# Tambahkan Garis Vertikal di x=16, x=50, x=84
vertical_lines_x = [16, 50, 84]
line_colors = {16: '#FF7F50', 50: '#FF4500', 84: '#FF6347'}
line_names = {16: 'x=16', 50: 'x=50', 84: 'x=84'}

for x_val in vertical_lines_x:
    fig.add_shape(
        type="line",
        x0=x_val, y0=y_values.min() * 0.9 if y_values.min() < 0 else 0,
        x1=x_val, y1=y_values.max() * 1.1,
        line=dict(color=line_colors[x_val], width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x_val, y=y_values.max() * 1.05, text=f"X={x_val}", showarrow=False,
        font=dict(color=line_colors[x_val], size=14, family="Montserrat, sans-serif", weight="bold")
    )

# Titik perpotongan kurva asli dengan x=16, 50, 84
if not np.isnan(results['y_at_x_16_original_curve']):
    fig.add_trace(go.Scatter(
        x=[16], y=[results['y_at_x_16_original_curve']],
        mode='markers',
        name=f"Y di X=16 (Asli): {results['y_at_x_16_original_curve']:.2f}",
        marker=dict(size=14, color=line_colors[16], symbol='circle', line=dict(width=2, color='white'))
    ))
if not np.isnan(results['y_at_x_50_original_curve']):
    fig.add_trace(go.Scatter(
        x=[50], y=[results['y_at_x_50_original_curve']],
        mode='markers',
        name=f"Y di X=50 (Asli): {results['y_at_x_50_original_curve']:.2f}",
        marker=dict(size=14, color=line_colors[50], symbol='circle', line=dict(width=2, color='white'))
    ))
if not np.isnan(results['y_at_x_84_original_curve']):
    fig.add_trace(go.Scatter(
        x=[84], y=[results['y_at_x_84_original_curve']],
        mode='markers',
        name=f"Y di X=84 (Asli): {results['y_at_x_84_original_curve']:.2f}",
        marker=dict(size=14, color=line_colors[84], symbol='circle', line=dict(width=2, color='white'))
    ))

# Kondisional untuk Garis Titik ke-10 & ke-20
if analysis_choice in ["Garis Titik ke-10 & ke-20", "Tampilkan Semua"]:
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']):
        fig.add_trace(go.Scatter(
            x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
            y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
            mode='markers', name='Titik Referensi (10 & 20)',
            marker=dict(size=12, color='#FFD700', symbol='star', line=dict(width=2, color='white'))
        ))
        fig.add_trace(go.Scatter(
            x=results['pt10_20_line_x_range'], y=results['pt10_20_line_y'],
            mode='lines', name='Garis 10 & 20',
            line=dict(color="#B8860B", width=3, dash="dot"), showlegend=True
        ))
        # Titik perpotongan untuk garis 10-20
        if not np.isnan(results['y_at_x_16_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[16], y=[results['y_at_x_16_pt10_20_line']],
                mode='markers', name=f'Y di X=16 (10-20): {results["y_at_x_16_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#B8860B', symbol='square-open', line=dict(width=3, color='#B8860B'))
            ))
        if not np.isnan(results['y_at_x_50_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[50], y=[results['y_at_x_50_pt10_20_line']],
                mode='markers', name=f'Y di X=50 (10-20): {results["y_at_x_50_pt10_20_line"]:.2f}',
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
                mode='markers', name=f'Y di X=84 (10-20): {results["y_at_x_84_pt10_20_line']:.2f}',
                marker=dict(size=14, color='#B8860B', symbol='triangle-up-open', line=dict(width=3, color='#B8860B'))
            ))

# Kondisional untuk Garis Regresi RANSAC
if analysis_choice in ["Garis RANSAC", "Tampilkan Semua"]:
    if not np.isnan(results['y_at_x_50_ransac_line']) and len(results['ransac_line_x']) > 0:
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'], y=results['ransac_line_y'],
            mode='lines', name='Garis RANSAC',
            line=dict(color='#00CED1', width=3, dash='dash'), showlegend=True
        ))
        # Titik perpotongan untuk garis RANSAC
        if not np.isnan(results['y_at_x_16_ransac_line']):
            fig.add_trace(go.Scatter(
                x=[16], y=[results['y_at_x_16_ransac_line']],
                mode='markers', name=f'Y di X=16 (RANSAC): {results["y_at_x_16_ransac_line"]:.2f}',
                marker=dict(size=14, color='#00CED1', symbol='diamond-open', line=dict(width=3, color='#00CED1'))
            ))
        if not np.isnan(results['y_at_x_50_ransac_line']):
            fig.add_trace(go.Scatter(
                x=[50], y=[results['y_at_x_50_ransac_line']],
                mode='markers', name=f'Y di X=50 (RANSAC): {results["y_at_x_50_ransac_line"]:.2f}',
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
                mode='markers', name=f'Y di X=84 (RANSAC): {results["y_at_x_84_ransac_line']:.2f}',
                marker=dict(size=14, color='#00CED1', symbol='triangle-up', line=dict(width=3, color='#00CED1'))
            ))

# Update layout untuk Plotly
fig.update_layout(
    xaxis_title="Siklus (x)",
    yaxis_title="Benang Putus (N)",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=12, color="#E0E0E0"), bgcolor="rgba(26,26,26,0.7)", borderwidth=1, bordercolor="#3A3A3A"
    ),
    margin=dict(l=20, r=20, t=60, b=20), height=600, template="plotly_dark",
    plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A", font=dict(color="#E0E0E0", family="Montserrat, sans-serif"),
    hoverlabel=dict(bgcolor="rgba(26,26,26,0.9)", font_size=14, font_family="Montserrat, sans-serif")
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#282828', zeroline=True, zerolinewidth=1.5, zerolinecolor='#282828', tickfont=dict(color="#B0B0B0"))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#282828', zeroline=True, zerolinewidth=1.5, zerolinecolor='#282828', tickfont=dict(color="#B0B0B0"))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Tampilkan Hasil Analisis (Sesuai dengan analysis_choice)
st.write("#### Hasil Nilai Penting (Perpotongan)")

st.markdown(f"""
<div class="dark-card" style="text-align: center; padding: 25px; margin-bottom: 20px;">
""", unsafe_allow_html=True)

if analysis_choice == "Kurva Asli":
    st.markdown(f"""
    <p style="color: #B0B0B0; font-size: 16px;">Hasil dari **Kurva Data Asli** Anda:</p>
    <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=16</p>
            <h2 style="color: {line_colors[16]}; font-size: 38px; margin: 5px 0;">{results['y_at_x_16_original_curve']:.2f}</h2>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=50</p>
            <h2 style="color: {line_colors[50]}; font-size: 38px; margin: 5px 0;">{results['y_at_x_50_original_curve']:.2f}</h2>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=84</p>
            <h2 style="color: {line_colors[84]}; font-size: 38px; margin: 5px 0;">{results['y_at_x_84_original_curve']:.2f}</h2>
        </div>
    </div>
    <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">Ini adalah nilai Y yang diinterpolasi langsung dari data mentah Anda.</div>
    """, unsafe_allow_html=True)

elif analysis_choice == "Garis Titik ke-10 & ke-20":
    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil dari **Garis Titik ke-10 & ke-20**:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=16</p>
                <h2 style="color: #B8860B; font-size: 38px; margin: 5px 0;">{results['y_at_x_16_pt10_20_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=50</p>
                <h2 style="color: #B8860B; font-size: 38px; margin: 5px 0;">{results['y_at_x_50_pt10_20_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=84</p>
                <h2 style="color: #B8860B; font-size: 38px; margin: 5px 0;">{results['y_at_x_84_pt10_20_line']:.2f}</h2>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">Dihitung dari garis lurus yang melewati titik data ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f}).</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Titik ke-10 & ke-20:</p>
        <h1 style="color: #B8860B; font-size: 60px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">
            Tidak cukup data untuk menghitung garis ini (minimal 20 titik diperlukan).
        </div>
        """, unsafe_allow_html=True)

elif analysis_choice == "Garis RANSAC":
    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil dari **Garis Regresi RANSAC**:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=16</p>
                <h2 style="color: #00CED1; font-size: 38px; margin: 5px 0;">{results['y_at_x_16_ransac_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=50</p>
                <h2 style="color: #00CED1; font-size: 38px; margin: 5px 0;">{results['y_at_x_50_ransac_line']:.2f}</h2>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Nilai Y di X=84</p>
                <h2 style="color: #00CED1; font-size: 38px; margin: 5px 0;">{results['y_at_x_84_ransac_line']:.2f}</h2>
            </div>
        </div>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">Ini adalah garis tren terbaik yang mengabaikan data outlier (pencilan), memberikan hasil yang lebih stabil.</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h1 style="color: #00CED1; font-size: 60px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 15px; color: #B0B0B0;">
            Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC (minimal 2 titik diperlukan).
        </div>
        """, unsafe_allow_html=True)

elif analysis_choice == "Tampilkan Semua":
    st.markdown(f"""
    <p style="color: #B0B0B0; font-size: 16px;">Hasil untuk **Kurva Data Asli**:</p>
    <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=16</p>
            <h3 style="color: {line_colors[16]}; font-size: 28px; margin: 5px 0;">{results['y_at_x_16_original_curve']:.2f}</h3>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=50</p>
            <h3 style="color: {line_colors[50]}; font-size: 28px; margin: 5px 0;">{results['y_at_x_50_original_curve']:.2f}</h3>
        </div>
        <div>
            <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=84</p>
            <h3 style="color: {line_colors[84]}; font-size: 28px; margin: 5px 0;">{results['y_at_x_84_original_curve']:.2f}</h3>
        </div>
    </div>
    <hr style="border-color: #3A3A3A !important; margin: 15px 0 !important;">
    """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil untuk **Garis Titik ke-10 & ke-20**:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=16</p>
                <h3 style="color: #B8860B; font-size: 28px; margin: 5px 0;">{results['y_at_x_16_pt10_20_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=50</p>
                <h3 style="color: #B8860B; font-size: 28px; margin: 5px 0;">{results['y_at_x_50_pt10_20_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=84</p>
                <h3 style="color: #B8860B; font-size: 28px; margin: 5px 0;">{results['y_at_x_84_pt10_20_line']:.2f}</h3>
            </div>
        </div>
        <hr style="border-color: #3A3A3A !important; margin: 15px 0 !important;">
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil untuk **Garis Titik ke-10 & ke-20**:</p>
        <h3 style="color: #B8860B; font-size: 28px; margin: 10px 0;">N/A</h3>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data untuk menghitung garis ini (minimal 20 titik diperlukan).</p>
        <hr style="border-color: #3A3A3A !important; margin: 15px 0 !important;">
        """, unsafe_allow_html=True)
    
    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil untuk **Garis Regresi RANSAC**:</p>
        <div style="display: flex; justify-content: space-around; margin-bottom: 10px;">
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=16</p>
                <h3 style="color: #00CED1; font-size: 28px; margin: 5px 0;">{results['y_at_x_16_ransac_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=50</p>
                <h3 style="color: #00CED1; font-size: 28px; margin: 5px 0;">{results['y_at_x_50_ransac_line']:.2f}</h3>
            </div>
            <div>
                <p style="color: #B0B0B0; font-size: 14px; margin: 0;">Y di X=84</p>
                <h3 style="color: #00CED1; font-size: 28px; margin: 5px 0;">{results['y_at_x_84_ransac_line']:.2f}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #B0B0B0; font-size: 16px;">Hasil untuk **Garis Regresi RANSAC**:</p>
        <h3 style="color: #00CED1; font-size: 28px; margin: 10px 0;">N/A</h3>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.</p>
        """, unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)

# --- Perhitungan SD dan CV ---
st.write("#### 3. Hasil Perhitungan SD & CV")

st.markdown("""
<div class="dark-card">
""", unsafe_allow_html=True)

if not np.isnan(results['sd_result']) and not np.isnan(results['cv_result']):
    st.markdown(f"""
    <p style="font-size: 18px; color: #DAA520; font-weight: 600;">Standar Deviasi (SD):</p>
    <p style="font-size: 22px; color: #E0E0E0; font-weight: 700;">
        $ SD = \\frac{{\\text{{Y di X=84}} - \\text{{Y di X=16}}}}{2} $
    </p>
    <p style="font-size: 26px; color: #00CED1; font-weight: 700;">
        $ SD = \\frac{{{results['y_at_x_84_ransac_line']:.2f} - {results['y_at_x_16_ransac_line']:.2f}}}{2} = {results['sd_result']:.2f} $
    </p>
    <br>
    <p style="font-size: 18px; color: #DAA520; font-weight: 600;">Koefisien Variasi (CV):</p>
    <p style="font-size: 22px; color: #E0E0E0; font-weight: 700;">
        $ CV = \\frac{{SD \\times 100}}{{\\text{{Y di X=50}}}} \\% $
    </p>
    <p style="font-size: 26px; color: #00CED1; font-weight: 700;">
        $ CV = \\frac{{{results['sd_result']:.2f} \\times 100}}{{{results['y_at_x_50_ransac_line']:.2f}}} = {results['cv_result']:.2f}\\% $
    </p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <p style="font-size: 18px; color: #B0B0B0;">
        Untuk menghitung SD dan CV, kami membutuhkan hasil nilai 'Benang Putus (N)' di siklus X=16, X=50, dan X=84 dari **Garis Regresi RANSAC**. <br>
        Pastikan Anda memiliki data yang cukup dan analisis RANSAC berhasil dilakukan.
    </p>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# --- Tampilan Tabel Data Lengkap ---
with st.expander("Lihat Semua Data yang Digunakan"):
    display_df_full = pd.DataFrame({
        'Siklus (x)': st.session_state.data['x_values'],
        'Benang Putus (N)': st.session_state.data['y_values']
    })
    display_df_full.index = display_df_full.index + 1
    st.dataframe(
        display_df_full,
        hide_index=False,
        use_container_width=True,
        height=400
    )

# --- Informasi & Footer ---
st.markdown("""
<div class="dark-card">
    <h3>Tentang Grafik Ini</h3>
    <p>Grafik ini menunjukkan bagaimana benang putus (N) berubah seiring bertambahnya siklus (x). Anda bisa memilih garis analisis mana yang ingin ditampilkan:</p>
    <ul>
        <li><strong style="color: #DAA520;">Kurva Asli:</strong> Menghubungkan titik data Anda secara langsung, menunjukkan tren dasar.</li>
        <li><strong style="color: #B8860B;">Garis Titik ke-10 & ke-20:</strong> Garis lurus yang ditarik antara titik ke-10 dan ke-20 dari data Anda. Metode ini kadang digunakan dalam standar tertentu.</li>
        <li><strong style="color: #00CED1;">Garis RANSAC:</strong> Ini adalah garis tren yang pintar. RANSAC bisa mengabaikan titik-titik data yang "aneh" (outlier) untuk menemukan pola utama, sehingga hasilnya lebih andal jika ada data yang tidak biasa.</li>
    </ul>
    <p>Titik-titik di X=16, X=50, dan X=84 adalah titik penting yang menunjukkan nilai Benang Putus (N) pada siklus tersebut, sesuai dengan garis yang Anda pilih.</p>
    <h3>Memahami SD & CV</h3>
    <p>
        <strong style="color: #DAA520;">Standar Deviasi (SD):</strong> Menunjukkan seberapa jauh nilai-nilai data tersebar dari rata-rata. SD yang lebih kecil berarti data lebih konsisten. Kami menghitungnya dari garis RANSAC: setengah dari selisih Y di X=84 dan X=16.
    </p>
    <p>
        <strong style="color: #DAA520;">Koefisien Variasi (CV):</strong> Mengukur tingkat keragaman data relatif terhadap rata-rata. CV yang rendah (<10%) biasanya menunjukkan data yang sangat konsisten, sementara nilai yang lebih tinggi menunjukkan variasi yang lebih besar. Kami menghitungnya dari garis RANSAC: SD dikalikan 100, lalu dibagi Y di X=50.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Benang Abrasi - Dibuat oleh RADIX <br>
    © 2025 Semua Hak Dilindungi.
</div>
""", unsafe_allow_html=True)
