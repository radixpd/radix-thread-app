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

# --- CSS Kustom untuk Tampilan Dark Mode Elegan (dengan Sentuhan Gold/Amber) ---
st.markdown("""
<style>
    /* Import Google Font - Playfair Display for headings */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Open+Sans:wght@300;400;600&display=swap');

    /* Color Palette */
    :root {
        --bg-primary: #0A0A0A; /* Very dark background */
        --bg-secondary: #1C1C1C; /* Slightly lighter for cards/tabs */
        --text-light: #F0F0F0; /* Main text color */
        --text-medium: #B0B0B0; /* Secondary text/subtle info */
        --accent-blue: #4F8EF7; /* Original accent blue, can still be used for highlights */
        --accent-gold: #FFD700; /* Gold/Amber accent */
        --accent-red: #E6341E; /* For x=50 line and original curve */
        --accent-orange: #FF5733; /* For 10-20 line */
        --accent-cyan: #00FFFF; /* For RANSAC line */
        --border-dark: #2A2A2A; /* Darker borders */
        --shadow-dark: rgba(0,0,0,0.4); /* Deeper shadows */
    }

    /* General Styles */
    .main {
        background-color: var(--bg-primary);
        color: var(--text-light);
        font-family: 'Open Sans', sans-serif;
    }
    .stApp {
        max-width: 1300px; /* Slightly wider */
        margin: 0 auto;
        padding-top: 30px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif; /* Elegant font for headings */
        letter-spacing: 1px;
        color: var(--text-light); /* Default heading color */
    }
    h1 {
        font-weight: 700;
        font-size: 42px;
        padding-bottom: 15px;
        border-bottom: 2px solid var(--accent-gold); /* Gold underline for main title */
        text-align: center;
        margin-bottom: 30px;
    }
    h2 {
        font-weight: 700;
        font-size: 30px;
        color: var(--accent-gold); /* Sub-headings in gold */
        margin-bottom: 20px;
        border-bottom: 1px solid var(--border-dark);
        padding-bottom: 10px;
        text-align: center; /* Center align subheaders */
    }
    h3 {
        font-weight: 600;
        font-size: 24px;
        color: var(--text-light);
        margin-top: 25px;
        margin-bottom: 15px;
    }
    p, li, span, label {
        color: var(--text-medium);
        font-family: 'Open Sans', sans-serif;
        line-height: 1.7;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--accent-gold); /* Gold button */
        color: var(--bg-primary); /* Dark text on gold */
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 12px 25px;
        font-size: 17px;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #FFEA8A; /* Lighter gold on hover */
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
        transform: translateY(-3px);
    }
    .stButton>button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(255, 215, 0, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Open Sans', sans-serif;
        color: var(--text-medium);
        font-weight: 600;
        padding: 15px 20px;
        font-size: 17px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--accent-gold); /* Gold text on hover */
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 12px;
        background-color: var(--bg-secondary);
        box-shadow: 0 6px 20px var(--shadow-dark);
        margin-bottom: 30px;
        border: 1px solid var(--border-dark);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 30px;
        border-radius: 12px;
        background-color: var(--bg-secondary);
        box-shadow: 0 6px 20px var(--shadow-dark);
        border: 1px solid var(--border-dark);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent-gold); /* Gold highlight */
        border-radius: 6px;
        height: 4px; /* Thicker highlight */
    }

    /* Radio Buttons - Unified for Graph & Results */
    .stRadio > label {
        color: var(--text-light);
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
        text-align: center; /* Center radio label */
        display: block; /* Make label a block for centering */
    }
    .stRadio > div { /* Container for radio buttons */
        background-color: var(--bg-secondary);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 20px var(--shadow-dark);
        display: flex;
        flex-wrap: wrap;
        gap: 15px; /* More space between items */
        border: 1px solid var(--border-dark);
    }
    .stRadio [data-baseweb="radio"] { /* Individual radio item */
        background-color: #282828; /* Slightly lighter dark for unselected */
        border-radius: 10px;
        padding: 12px 20px;
        transition: background-color 0.3s ease, border 0.3s ease;
        flex-grow: 1;
        text-align: center;
        border: 1px solid #3A3A3A; /* Subtle border */
    }
    .stRadio [data-baseweb="radio"]:hover {
        background-color: #383838;
        border-color: var(--accent-gold); /* Gold border on hover */
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: var(--accent-gold) !important; /* Gold background when selected */
        color: var(--bg-primary); /* Dark text on gold */
        border: 1px solid var(--accent-gold);
    }
    .stRadio [data-baseweb="radio"] span:last-child { /* text of the radio button */
        color: var(--text-light); /* Default text color */
        font-weight: 600;
        font-size: 16px;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: var(--bg-primary); /* Dark text when selected */
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: var(--bg-secondary);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 25px var(--shadow-dark);
        margin-bottom: 30px;
        border: 1px solid var(--border-dark);
    }

    /* Radix Header */
    .app-header {
        background-color: var(--bg-secondary);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 35px var(--shadow-dark);
        margin-bottom: 40px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(8px);
        border: 1px solid var(--border-dark);
    }
    .pulcra-logo {
        font-family: 'Playfair Display', serif; /* Match heading font */
        font-weight: 700;
        font-size: 52px; /* Larger logo */
        color: var(--accent-gold); /* Gold logo */
        margin-bottom: 10px;
        letter-spacing: 4px;
        text-shadow: 0 4px 20px rgba(255, 215, 0, 0.6); /* Stronger gold shadow */
    }
    .app-header h1 {
        font-family: 'Open Sans', sans-serif; /* Keep this simpler */
        font-size: 36px;
        color: var(--text-light);
        margin-top: 15px;
        border-bottom: none;
        padding-bottom: 0;
    }
    .app-header p {
        color: var(--text-medium);
        font-size: 18px;
        text-align: center;
        margin-top: 10px;
    }

    /* Footer */
    .radix-footer {
        text-align: center;
        margin-top: 60px;
        padding: 25px;
        font-size: 15px;
        font-family: 'Open Sans', sans-serif;
        color: var(--text-medium);
        border-top: 1px solid var(--border-dark);
        background-color: var(--bg-secondary);
        border-radius: 0 0 15px 15px;
        box-shadow: 0 -4px 15px var(--shadow-dark);
    }

    /* Other elements */
    hr {
        border-color: var(--border-dark) !important;
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
        border: 1px solid var(--border-dark);
        box-shadow: 0 4px 15px var(--shadow-dark);
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 3px dashed var(--accent-gold); /* Gold dashed border */
        border-radius: 12px;
        padding: 30px;
        background-color: #1A1A1A; /* Slightly lighter background */
        box-shadow: 0 4px 15px var(--shadow-dark);
    }
    [data-testid="stFileUploaderDropzone"] p {
        color: var(--text-medium);
        font-size: 16px;
    }

    /* For data preview tables */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 15px var(--shadow-dark);
        background-color: var(--bg-secondary);
        max-height: 350px; /* Limit height to enable scrolling */
        overflow-y: auto; /* Enable vertical scroll */
        border: 1px solid var(--border-dark); /* subtle border */
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
    }
    .stDataFrame th {
        background-color: #282828 !important; /* Slightly lighter header for contrast */
        color: var(--accent-gold) !important; /* Gold header text */
        font-weight: 700;
        position: sticky;
        top: 0;
        z-index: 1;
        padding: 15px 10px; /* More padding */
    }
    .stDataFrame td {
        background-color: var(--bg-secondary) !important;
        color: var(--text-light) !important;
        border-bottom: 1px solid var(--border-dark) !important;
        padding: 12px 10px; /* More padding */
    }
    /* Scrollbar for dataframes */
    .stDataFrame::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    .stDataFrame::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    .stDataFrame::-webkit-scrollbar-thumb {
        background: var(--accent-gold); /* Gold scrollbar */
        border-radius: 10px;
    }
    .stDataFrame::-webkit-scrollbar-thumb:hover {
        background: #FFEA8A; /* Lighter gold on hover */
    }

    /* Text input and number input styles */
    .stTextInput > div > div > input, .stNumberInput > div > input {
        background-color: #1A1A1A; /* Darker input fields */
        color: var(--text-light);
        border: 1px solid var(--border-dark);
        border-radius: 8px;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > input:focus {
        border-color: var(--accent-gold); /* Gold border on focus */
        box-shadow: 0 0 0 0.2rem rgba(255, 215, 0, 0.25);
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
        password_input = st.text_input("Masukkan Kode Akses Anda", type="password", key="password_input")
        col_pw1, col_pw2, col_pw3 = st.columns([1,2,1]) # Wider middle column for button
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
    <h1 style="margin-top: 0; color: var(--text-light); font-size: 36px; border-bottom: none; padding-bottom: 0;">Analisis Abrasi Benang</h1>
    <p style="color: var(--text-medium); font-size: 18px; text-align: center;">Alat profesional untuk visualisasi data dan perhitungan nilai perpotongan</p>
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
            # st.warning(f"Error Regresi RANSAC: {e}. Pastikan data memiliki variasi.") # Suppress this warning if it's too frequent
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])
    # else:
        # st.warning("Data tidak cukup untuk Regresi RANSAC (minimal 2 titik).") # Suppress this warning

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
    line=dict(color=st.get_option("theme.primaryColor"), width=3), # Use Streamlit's primary color
    marker=dict(size=8, color=st.get_option("theme.primaryColor"))
))

# Tambahkan Garis Vertikal di x=50 (selalu)
fig.add_shape(
    type="line",
    x0=50, y0=y_values.min() * 0.9 if y_values.min() < 0 else 0,
    x1=50, y1=y_values.max() * 1.1,
    line=dict(color="#FFD700", width=2, dash="dash"), # Gold dashed line
)
# Adjusted annotation position for better visibility if lines overlap
fig.add_annotation(
    x=50, y=y_values.max() * 1.05, text="x=50", showarrow=False,
    font=dict(color="#FFD700", size=14, family="Open Sans, sans-serif", weight="bold")
)

# Tambahkan titik perpotongan kurva asli dengan x=50 (selalu)
if not np.isnan(results['y_at_x_50_original_curve']):
    fig.add_trace(go.Scatter(
        x=[50], y=[results['y_at_x_50_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=50, y={results["y_at_x_50_original_curve"]:.2f}',
        marker=dict(size=14, color=st.get_option("theme.primaryColor"), symbol='circle', line=dict(width=2, color='white'))
    ))

# Kondisional untuk Garis Titik 10 & 20
if analysis_choice in ["Garis Titik 10 & 20", "Tampilkan Semua"]:
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']):
        fig.add_trace(go.Scatter(
            x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
            y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
            mode='markers', name='Titik Referensi (10 & 20)',
            marker=dict(size=12, color='#FFEA8A', symbol='star', line=dict(width=2, color='white')) # Lighter gold star
        ))
        fig.add_trace(go.Scatter(
            x=results['pt10_20_line_x_range'], y=results['pt10_20_line_y'],
            mode='lines', name='Garis Titik 10 & 20',
            line=dict(color="#DAA520", width=3, dash="dot"), showlegend=True # Darker gold/bronze
        ))
        if not np.isnan(results['y_at_x_50_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[50], y=[results['y_at_x_50_pt10_20_line']],
                mode='markers', name=f'Int. Garis 10-20 di x=50, y={results["y_at_x_50_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#DAA520', symbol='circle-open', line=dict(width=3, color='#DAA520'))
            ))
            y_pos_pt10_20_label = results['y_at_x_50_pt10_20_line'] + (y_values.max() * 0.05 if y_values.max() > 0 else 50)
            fig.add_annotation(
                x=50, y=y_pos_pt10_20_label, text=f"Garis 10-20: {results['y_at_x_50_pt10_20_line']:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#DAA520',
                font=dict(size=14, color='#DAA520', family="Open Sans, sans-serif"),
                bordercolor="#DAA520", borderwidth=1, borderpad=4, bgcolor="rgba(30,30,30,0.7)", opacity=0.9
            )

# Kondisional untuk Garis Regresi RANSAC
if analysis_choice in ["Garis Regresi RANSAC", "Tampilkan Semua"]:
    if not np.isnan(results['y_at_x_50_ransac_line']) and len(results['ransac_line_x']) > 0:
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'], y=results['ransac_line_y'],
            mode='lines', name='Regresi RANSAC',
            line=dict(color='#8A2BE2', width=3, dash='dash'), showlegend=True # Royal Purple for RANSAC
        ))
        fig.add_trace(go.Scatter(
            x=[50], y=[results['y_at_x_50_ransac_line']],
            mode='markers', name=f'Int. RANSAC di x=50, y={results["y_at_x_50_ransac_line']:.2f}',
            marker=dict(size=14, color='#8A2BE2', symbol='diamond-open', line=dict(width=3, color='#8A2BE2'))
        ))
        y_pos_ransac_label = results['y_at_x_50_ransac_line'] - (y_values.max() * 0.05 if results['y_at_x_50_ransac_line'] > 0 else 50)
        fig.add_annotation(
            x=50, y=y_pos_ransac_label, text=f"RANSAC: {results['y_at_x_50_ransac_line']:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#8A2BE2',
            font=dict(size=14, color='#8A2BE2', family="Open Sans, sans-serif"),
            bordercolor="#8A2BE2", borderwidth=1, borderpad=4, bgcolor="rgba(30,30,30,0.7)", opacity=0.9
        )

# Update layout
fig.update_layout(
    xaxis_title="Nilai Tetap (x)",
    yaxis_title="Nilai Benang Putus (N)",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=12, color="var(--text-medium)"), bgcolor="rgba(28,28,28,0.8)", borderwidth=1, bordercolor="var(--border-dark)"
    ),
    margin=dict(l=10, r=10, t=50, b=10), height=550, template="plotly_dark",
    plot_bgcolor="var(--bg-secondary)", paper_bgcolor="var(--bg-secondary)", font=dict(color="var(--text-light)")
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='var(--border-dark)', zeroline=True, zerolinewidth=1.5, zerolinecolor='var(--border-dark)', tickfont=dict(color="var(--text-medium)"))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='var(--border-dark)', zeroline=True, zerolinewidth=1.5, zerolinecolor='var(--border-dark)', tickfont=dict(color="var(--text-medium)"))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---") # Garis pemisah untuk hasil

# Tampilkan Hasil Analisis (Sesuai dengan analysis_choice)
st.write("#### Hasil Nilai Perpotongan pada x=50")

st.markdown(f"""
<div class="dark-card" style="text-align: center; padding: 25px; margin-bottom: 20px;">
""", unsafe_allow_html=True)

if analysis_choice == "Kurva Data Asli":
    val = results['y_at_x_50_original_curve']
    desc = "Nilai perpotongan kurva data abrasi asli pada x=50. Ini adalah interpolasi langsung dari data yang Anda masukkan."
    st.markdown(f"""
    <p style="color: var(--text-medium); font-size: 16px;">Hasil Kurva Data Asli:</p>
    <h1 style="color: {st.get_option("theme.primaryColor")}; font-size: 48px; margin: 10px 0;">{val:.2f}</h1>
    <div style="margin-top: 15px; font-size: 14px; color: var(--text-medium);">{desc}</div>
    """, unsafe_allow_html=True)

elif analysis_choice == "Garis Titik 10 & 20":
    val = results['y_at_x_50_pt10_20_line']
    if not np.isnan(val):
        desc = f"Berdasarkan garis linear yang ditarik antara titik data ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik data ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f})."
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <h1 style="color: #DAA520; font-size: 48px; margin: 10px 0;">{val:.2f}</h1>
        <div style="margin-top: 15px; font-size: 14px; color: var(--text-medium);">{desc}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <h1 style="color: #DAA520; font-size: 48px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 14px; color: var(--text-medium);">
            Tidak cukup data untuk menghitung garis ini (membutuhkan setidaknya 20 titik).
        </div>
        """, unsafe_allow_html=True)

elif analysis_choice == "Garis Regresi RANSAC":
    val = results['y_at_x_50_ransac_line']
    if not np.isnan(val):
        desc = "Berdasarkan model Regresi Linear Robust (RANSAC), yang secara cerdas menemukan garis terbaik dengan mengabaikan data outlier."
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h1 style="color: #8A2BE2; font-size: 48px; margin: 10px 0;">{val:.2f}</h1>
        <div style="margin-top: 15px; font-size: 14px; color: var(--text-medium);">{desc}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h1 style="color: #8A2BE2; font-size: 48px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 14px; color: var(--text-medium);">
            Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.
        </div>
        """, unsafe_allow_html=True)

elif analysis_choice == "Tampilkan Semua":
    st.markdown(f"""
    <p style="color: var(--text-medium); font-size: 16px;">Hasil Kurva Data Asli:</p>
    <h1 style="color: {st.get_option("theme.primaryColor")}; font-size: 48px; margin: 10px 0;">{results['y_at_x_50_original_curve']:.2f}</h1>
    <p style="color: var(--text-medium); font-size: 14px; margin-bottom: 20px;">Nilai perpotongan kurva data abrasi asli pada x=50.</p>
    """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <h1 style="color: #DAA520; font-size: 48px; margin: 10px 0;">{results['y_at_x_50_pt10_20_line']:.2f}</h1>
        <p style="color: var(--text-medium); font-size: 14px; margin-bottom: 20px;">Berdasarkan garis linear antara titik data ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik data ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f}).</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Titik 10 & 20:</p>
        <h1 style="color: #DAA520; font-size: 48px; margin: 10px 0;">N/A</h1>
        <p style="color: var(--text-medium); font-size: 14px; margin-bottom: 20px;">Tidak cukup data untuk menghitung garis ini (membutuhkan setidaknya 20 titik).</p>
        """, unsafe_allow_html=True)
    
    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h1 style="color: #8A2BE2; font-size: 48px; margin: 10px 0;">{results['y_at_x_50_ransac_line']:.2f}</h1>
        <p style="color: var(--text-medium); font-size: 14px; margin-bottom: 20px;">Berdasarkan model Regresi Linear Robust (RANSAC) yang mempertimbangkan outlier.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: var(--text-medium); font-size: 16px;">Hasil Garis Regresi RANSAC:</p>
        <h1 style="color: #8A2BE2; font-size: 48px; margin: 10px 0;">N/A</h1>
        <p style="color: var(--text-medium); font-size: 14px; margin-bottom: 20px;">Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.</p>
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
        <li><strong style="color: {st.get_option("theme.primaryColor")};">Kurva Data Asli:</strong> Garis biru mewakili data abrasi benang yang Anda masukkan, diinterpolasi untuk menunjukkan tren.</li>
        <li><strong style="color: #FFEA8A;">Titik Referensi (10 & 20):</strong> Bintang emas menandai titik data ke-10 dan ke-20. Titik-titik ini digunakan sebagai referensi untuk "Garis Titik 10 & 20".</li>
        <li><strong style="color: #DAA520;">Garis Titik 10 & 20:</strong> Garis putus-putus berwarna emas tua adalah proyeksi linear yang ditarik antara titik data ke-10 dan ke-20, diekstrapolasi hingga x=50.</li>
        <li><strong style="color: #8A2BE2;">Garis Regresi RANSAC:</strong> Garis putus-putus ungu adalah model regresi linear robust yang mengidentifikasi dan mengabaikan titik data 'outlier' (pencilan), sehingga memberikan garis tren yang lebih stabil.</li>
        <li><strong style="color: #FFD700;">Garis Vertikal x=50:</strong> Garis putus-putus emas menunjukkan titik referensi utama pada sumbu-x (nilai 50), di mana nilai benang putus dihitung.</li>
    </ul>
    <h3>Tips Interaksi</h3>
    <ul>
        <li>Arahkan kursor ke titik atau garis pada grafik untuk melihat detail nilainya.</li>
        <li>Klik dan seret di grafik untuk memperbesar area tertentu (zoom in).</li>
        <li>Klik dua kali pada grafik untuk mengatur ulang tampilan ke skala awal (reset zoom).</li>
        <li>Gunakan toolbar di kanan atas grafik untuk opsi interaksi lainnya (pan, zoom, save as image).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="radix-footer">
    Aplikasi Analisis Abrasi Benang - Dibuat oleh Radix Team © 2025
</div>
""", unsafe_allow_html=True)
