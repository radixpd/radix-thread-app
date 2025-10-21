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

# --- CSS Kustom untuk Tampilan Light Mode Penuh & Elegan ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Playfair+Display:wght@400;700&display=swap');

    /* 💡 Perubahan Utama ke Light Mode: Warna Latar Belakang & Teks */
    html, body {
        background-color: #FFFFFF !important; /* Putih Cerah */
        color: #333333 !important; /* Teks Gelap */
    }

    /* Streamlit's main wrapper */
    .stApp {
        background-color: #FFFFFF !important; /* Putih */
        max-width: 1300px;
        margin: 0 auto;
        padding-top: 30px;
        padding-bottom: 50px;
        padding-left: 15px;
        padding-right: 15px;
    }
    
    /* Main content area within .stApp */
    .main {
        background-color: #FFFFFF; /* Putih */
        color: #333333; /* Teks Gelap */
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: #F8F8F8 !important; /* Abu-abu Sangat Terang */
        color: #333333;
        border-right: 1px solid #DDDDDD; /* Garis pemisah terang */
    }
    
    .block-container {
        background-color: #FFFFFF !important; /* Putih */
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important; /* Hitam */
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.8px;
    }
    h1 {
        font-weight: 700;
        font-size: 44px;
        padding-bottom: 15px;
        border-bottom: 3px solid #6B4226; /* Cokelat Gelap */
        text-align: center;
        text-shadow: 0 4px 10px rgba(0,0,0,0.05); /* Bayangan Ringan */
    }
    h2 {
        font-weight: 600;
        font-size: 32px;
        color: #6B4226 !important; /* Cokelat Gelap untuk Heading */
        margin-bottom: 20px;
        border-bottom: 1px solid #EEEEEE; /* Garis Bawah Sangat Terang */
        padding-bottom: 8px;
        font-family: 'Montserrat', sans-serif;
    }
    p, li, span, div {
        color: #333333 !important; /* Teks Gelap */
        font-family: 'Montserrat', sans-serif;
        line-height: 1.8;
        font-size: 17px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #6B4226; /* Cokelat Gelap */
        color: white !important;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        background-color: #8B5A2B; /* Cokelat Sedikit Lebih Terang */
        box-shadow: 0 8px 25px rgba(107, 66, 38, 0.3);
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #333333 !important; /* Teks Gelap */
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F8F8F8; /* Abu-abu Sangat Terang */
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        border: 1px solid #DDDDDD;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #FFFFFF; /* Putih */
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        border: 1px solid #DDDDDD;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #6B4226; /* Cokelat Gelap */
    }

    /* Radio Buttons (Tetap elegan di Light Mode) */
    .stRadio > div {
        background-color: #F8F8F8; /* Abu-abu Sangat Terang */
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }
    .stRadio [data-baseweb="radio"] {
        background-color: #FFFFFF; /* Putih */
        border: 1px solid #DDDDDD;
    }
    .stRadio [data-baseweb="radio"]:hover {
        background-color: #F0F0F0; /* Abu-abu Saat Hover */
        border: 1px solid #6B4226;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: #6B4226 !important; /* Cokelat Gelap Saat Terpilih */
        color: #FFFFFF !important;
    }
    .stRadio [data-baseweb="radio"] span:last-child {
        color: #333333 !important; /* Teks Gelap */
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: #FFFFFF !important; /* Teks Putih Saat Terpilih */
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #F8F8F8; /* Abu-abu Sangat Terang */
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 6px 25px rgba(0,0,0,0.08);
        border: 1px solid #EEEEEE;
    }
    .result-card p {
        color: #333333 !important; /* Teks Gelap */
    }

    /* Header */
    .app-header {
        background: linear-gradient(145deg, #F8F8F8, #FFFFFF); /* Gradient Terang */
        border: 1px solid #DDDDDD;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
    .pulcra-logo {
        color: #6B4226; /* Cokelat Gelap */
        text-shadow: 0 5px 20px rgba(107, 66, 38, 0.1);
    }
    .app-header h1 {
        color: #333333; /* Teks Gelap */
    }
    .app-header p {
        color: #666666; /* Abu-abu Gelap */
    }

    /* Footer */
    .radix-footer {
        color: #666666; /* Abu-abu Gelap */
        border-top: 1px solid #DDDDDD;
        background-color: #F8F8F8; /* Abu-abu Sangat Terang */
    }

    /* Data Editor & File Uploader */
    [data-testid="stDataEditor"], .stDataFrame {
        border: 1px solid #DDDDDD; /* Garis Terang */
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        background-color: #FFFFFF; /* Putih */
    }
    [data-testid="stDataEditor"] .st-emotion-cache-16txt4s, [data-testid="stDataEditor"] .st-emotion-cache-zt5ig8 {
        /* Mempertahankan warna error/success yang bagus di light mode */
        color: #333333 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #6B4226; /* Cokelat Gelap */
        background-color: #F8F8F8; /* Abu-abu Sangat Terang */
    }
    .stDataFrame th {
        background-color: #EEEEEE !important; /* Header Tabel Abu-abu Terang */
        color: #6B4226 !important; /* Teks Header Cokelat Gelap */
    }
    .stDataFrame td {
        background-color: #FFFFFF !important; /* Sel Putih */
        color: #333333 !important; /* Teks Sel Gelap */
        border-bottom: 1px solid #F0F0F0 !important; /* Garis antar sel sangat terang */
    }
    .stDataFrame::-webkit-scrollbar-track {
        background: #F0F0F0; /* Abu-abu Terang */
    }
    .stDataFrame::-webkit-scrollbar-thumb {
        background: #6B4226; /* Cokelat Gelap */
    }
    
    /* Plotly specifics - Modebar */
    .js-plotly-plot .plotly .modebar {
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .js-plotly-plot .plotly .modebar-btn {
        color: #6B4226 !important; /* Ikon Modebar Cokelat Gelap */
    }
    .js-plotly-plot .plotly .modebar-btn:hover {
        background-color: #F0F0F0 !important;
    }

    /* Text Input */
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        border: 1px solid #DDDDDD;
        color: #333333;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Implementasi Kode Akses ---
ACCESS_CODE = "RADIX2025"
TARGET_X_VALUE = 50 

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

# --- Inisialisasi Session State ---
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

    x_np = x_values_series.values
    y_np = y_values_series.values

    if len(x_np) < 2 or len(y_np) < 2:
        # st.warning("Data tidak cukup untuk analisis. Masukkan minimal 2 pasangan X dan Y.") # Komentar ini bisa dihapus dari fungsi utama
        return results

    # Original curve interpolation
    try:
        if not np.all(np.diff(x_np) > 0):
            # st.error("Nilai 'x_values' harus monoton meningkat untuk interpolasi kurva.") # Komentar ini bisa dihapus
            return results
        f = interpolate.interp1d(x_np, y_np, kind='linear', fill_value='extrapolate')
        results['y_at_x_50_original_curve'] = float(f(TARGET_X_VALUE))
    except ValueError:
        pass

    # Garis Antara Titik 10 & 20
    if len(x_np) >= 20:
        results['specific_x1_pt10_20'] = x_np[9]
        results['specific_y1_pt10_20'] = y_np[9]
        results['specific_x2_pt10_20'] = x_np[19]
        results['specific_y2_pt10_20'] = y_np[19]
    elif len(x_np) >= 2:
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
            results['y_at_x_50_pt10_20_line'] = np.nan
            results['pt10_20_line_x_range'] = np.array([])
            results['pt10_20_line_y'] = np.array([])

    # Regresi Linear Robust (RANSAC)
    if len(x_np) >= 2:
        try:
            X_reshaped = x_np.reshape(-1, 1)
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
            
        except Exception:
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
            line=dict(color='#6B4226', width=3),  # 💡 Cokelat Gelap
            marker=dict(size=8, color='#8B5A2B') 
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
            line=dict(color="#FF6347", width=2, dash="dash"),
            layer="below"
        )
        fig.add_annotation(
            x=TARGET_X_VALUE, y=y1_line * 0.95,
            text=f"x={TARGET_X_VALUE}", showarrow=False,
            font=dict(color="#FF6347", size=14, family="Montserrat, sans-serif", weight="bold"),
            bgcolor="rgba(255,255,255,0.7)", bordercolor="#FF6347", borderwidth=1, borderpad=4
        )

        # Add specific lines based on exact choice
        if analysis_choice == "Garis Titik 10 & 20":
            if results.get('pt10_20_line_x_range', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['pt10_20_line_x_range'],
                    y=results['pt10_20_line_y'],
                    mode='lines',
                    name='Garis Titik 10 & 20',
                    line=dict(color='#007BFF', width=2, dash='dot')
                ))
                if not np.isnan(results['specific_x1_pt10_20']):
                    fig.add_trace(go.Scatter(x=[results['specific_x1_pt10_20']], y=[results['specific_y1_pt10_20']], mode='markers', name='Titik ke-10', marker=dict(size=10, color='#007BFF', symbol='circle')))
                if not np.isnan(results['specific_x2_pt10_20']):
                    fig.add_trace(go.Scatter(x=[results['specific_x2_pt10_20']], y=[results['specific_y2_pt10_20']], mode='markers', name='Titik ke-20', marker=dict(size=10, color='#007BFF', symbol='circle')))
            if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
                fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_pt10_20_line']], mode='markers', name='Potongan Garis 10-20 di x=50', marker=dict(size=12, color='#007BFF', symbol='star'), hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))
        
        elif analysis_choice == "Garis yang melewati banyak titik":
            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['ransac_line_x'],
                    y=results['ransac_line_y'],
                    mode='lines',
                    name='Regresi RANSAC',
                    line=dict(color='#28A745', width=2, dash='dash')
                ))
            if not np.isnan(results.get('y_at_x_50_ransac_line')):
                fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_ransac_line']], mode='markers', name='Potongan RANSAC di x=50', marker=dict(size=12, color='#28A745', symbol='star'), hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))

        elif analysis_choice == "Kurva Data Asli":
            if not np.isnan(results.get('y_at_x_50_original_curve')):
                 fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_original_curve']], mode='markers', name='Potongan Kurva Asli di x=50', marker=dict(size=12, color='#6B4226', symbol='star'), hovertemplate=f"<b>Potongan (Kurva Asli)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))
        
        elif analysis_choice == "Tampilkan Semua":
            if results.get('pt10_20_line_x_range', []).size > 0:
                fig.add_trace(go.Scatter(x=results['pt10_20_line_x_range'], y=results['pt10_20_line_y'], mode='lines', name='Garis Titik 10 & 20', line=dict(color='#007BFF', width=2, dash='dot')))
                if not np.isnan(results['specific_x1_pt10_20']): fig.add_trace(go.Scatter(x=[results['specific_x1_pt10_20']], y=[results['specific_y1_pt10_20']], mode='markers', name='Titik ke-10', marker=dict(size=10, color='#007BFF', symbol='circle')))
                if not np.isnan(results['specific_x2_pt10_20']): fig.add_trace(go.Scatter(x=[results['specific_x2_pt10_20']], y=[results['specific_y2_pt10_20']], mode='markers', name='Titik ke-20', marker=dict(size=10, color='#007BFF', symbol='circle')))
                if not np.isnan(results.get('y_at_x_50_pt10_20_line')): fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_pt10_20_line']], mode='markers', name='Potongan Garis 10-20 di x=50', marker=dict(size=12, color='#007BFF', symbol='star'), hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))

            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(x=results['ransac_line_x'], y=results['ransac_line_y'], mode='lines', name='Regresi RANSAC', line=dict(color='#28A745', width=2, dash='dash')))
            if not np.isnan(results.get('y_at_x_50_ransac_line')): fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_ransac_line']], mode='markers', name='Potongan RANSAC di x=50', marker=dict(size=12, color='#28A745', symbol='star'), hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))
            
            if not np.isnan(results.get('y_at_x_50_original_curve')): fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results['y_at_x_50_original_curve']], mode='markers', name='Potongan Kurva Asli di x=50', marker=dict(size=12, color='#6B4226', symbol='star'), hovertemplate=f"<b>Potongan (Kurva Asli)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}}<extra></extra>"))

    # 💡 MODIFIKASI PLOTLY UNTUK LIGHT MODE
    fig.update_layout(
        title={'text': 'Grafik Abrasi Benang', 'font': dict(color='#333333', size=24, family='Playfair Display, serif')},
        xaxis_title='Nilai X',
        yaxis_title='Nilai Benang Putus (N)',
        plot_bgcolor='#FFFFFF', # 💡 PUTIH
        paper_bgcolor='#FFFFFF', # 💡 PUTIH
        font=dict(color='#333333', family='Montserrat, sans-serif'), # 💡 TEKS GELAP
        xaxis=dict(showgrid=True, gridcolor='#DDDDDD', zeroline=False, title_font=dict(size=18), tickfont=dict(size=14)),
        yaxis=dict(showgrid=True, gridcolor='#DDDDDD', zeroline=False, title_font=dict(size=18), tickfont=dict(size=14)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#DDDDDD", borderwidth=1, 
            font=dict(size=14, color='#333333') # 💡 TEKS LEGEND GELAP
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, b=40, t=100)
    )
    
    # Add results annotations (Diubah warnanya agar terlihat di latar putih)
    if analysis_choice == "Kurva Data Asli" and not np.isnan(results.get('y_at_x_50_original_curve')):
        fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=f"<b>Kurva Asli:</b> {results['y_at_x_50_original_curve']:.2f} N", showarrow=False, font=dict(size=14, color="#6B4226"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#6B4226", borderwidth=1, borderpad=4)
    elif analysis_choice == "Garis Titik 10 & 20" and not np.isnan(results.get('y_at_x_50_pt10_20_line')):
        fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=f"<b>Garis 10-20:</b> {results['y_at_x_50_pt10_20_line']:.2f} N", showarrow=False, font=dict(size=14, color="#007BFF"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#007BFF", borderwidth=1, borderpad=4)
    elif analysis_choice == "Garis yang melewati banyak titik" and not np.isnan(results.get('y_at_x_50_ransac_line')):
        fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text=f"<b>RANSAC:</b> {results['y_at_x_50_ransac_line']:.2f} N", showarrow=False, font=dict(size=14, color="#28A745"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#28A745", borderwidth=1, borderpad=4)
    elif analysis_choice == "Tampilkan Semua":
        y_pos = 0.95
        if not np.isnan(results.get('y_at_x_50_original_curve')):
            fig.add_annotation(x=0.05, y=y_pos, xref="paper", yref="paper", text=f"<b>Kurva Asli:</b> {results['y_at_x_50_original_curve']:.2f} N", showarrow=False, font=dict(size=14, color="#6B4226"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#6B4226", borderwidth=1, borderpad=4)
            y_pos -= 0.08
        if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
            fig.add_annotation(x=0.05, y=y_pos, xref="paper", yref="paper", text=f"<b>Garis 10-20:</b> {results['y_at_x_50_pt10_20_line']:.2f} N", showarrow=False, font=dict(size=14, color="#007BFF"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#007BFF", borderwidth=1, borderpad=4)
            y_pos -= 0.08
        if not np.isnan(results.get('y_at_x_50_ransac_line')):
            fig.add_annotation(x=0.05, y=y_pos, xref="paper", yref="paper", text=f"<b>RANSAC:</b> {results['y_at_x_50_ransac_line']:.2f} N", showarrow=False, font=dict(size=14, color="#28A745"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#28A745", borderwidth=1, borderpad=4)

    return fig


# --- Fungsi Utama Aplikasi ---
def main_app():
    
    st.markdown("<h2>Upload Data Abrasi</h2>", unsafe_allow_html=True)

    with st.expander("Unggah atau Edit Data", expanded=True):
        col_up1, col_up2 = st.columns([2, 3])

        with col_up1:
            uploaded_file = st.file_uploader("Unggah file CSV atau Excel (.xlsx) Anda", type=['csv', 'xlsx'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_uploaded = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df_uploaded = pd.read_excel(uploaded_file)
                    
                    if 'x_values' in df_uploaded.columns and 'y_values' in df_uploaded.columns:
                        st.session_state.data = df_uploaded[['x_values', 'y_values']].astype(float)
                        st.session_state.data_needs_recalc = True
                        st.success("Data berhasil diunggah!")
                    else:
                        st.error("File harus memiliki kolom 'x_values' dan 'y_values'.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membaca file: {e}")
            
            if st.button("Reset Data ke Nilai Awal", key="reset_button"):
                st.session_state.data = pd.DataFrame(INITIAL_DATA)
                st.session_state.data_needs_recalc = True
                st.info("Data telah direset ke nilai awal.")

        with col_up2:
            st.markdown("<h3>Edit Data Langsung</h3>", unsafe_allow_html=True)
            
            edited_df = st.data_editor(
                st.session_state.data,
                column_config={
                    "x_values": st.column_config.NumberColumn("Cycles", format="%.2f"),
                    "y_values": st.column_config.NumberColumn("Tensile Strength (N)", format="%.0f"),
                },
                num_rows="dynamic",
                use_container_width=True
            )

            if not edited_df.equals(st.session_state.data):
                try:
                    cleaned_df = edited_df.dropna(subset=['x_values', 'y_values']).copy()
                    cleaned_df['x_values'] = pd.to_numeric(cleaned_df['x_values'], errors='coerce')
                    cleaned_df['y_values'] = pd.to_numeric(cleaned_df['y_values'], errors='coerce')
                    cleaned_df = cleaned_df.dropna()
                    
                    if len(cleaned_df) >= 2:
                        st.session_state.data = cleaned_df[['x_values', 'y_values']].reset_index(drop=True)
                        st.session_state.data_needs_recalc = True
                    elif len(cleaned_df) == 0:
                        st.warning("Data kosong. Masukkan minimal 2 baris data.")
                        st.session_state.data = pd.DataFrame(columns=['x_values', 'y_values'])
                    else:
                        st.warning("Data kurang dari 2 baris. Masukkan minimal 2 baris data.")
                        st.session_state.data = cleaned_df[['x_values', 'y_values']].reset_index(drop=True)
                except Exception as e:
                    st.error(f"Kesalahan pemrosesan data: Pastikan semua input adalah angka. {e}")
                    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if st.session_state.data_needs_recalc and len(st.session_state.data) >= 2:
        st.session_state.calculated_results = calculate_lines_and_points(
            st.session_state.data['x_values'], 
            st.session_state.data['y_values']
        )
        st.session_state.data_needs_recalc = False
    
    results = st.session_state.calculated_results

    st.markdown("<h2>Hasil Analisis & Visualisasi</h2>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Visualisasi Grafik", "Ringkasan Hasil"])

    with tab1:
        st.markdown("<h3>Pilihan Tampilan Garis Analisis</h3>", unsafe_allow_html=True)
        analysis_choice = st.radio(
            "Pilih metode analisis yang ingin ditampilkan pada grafik:",
            ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis yang melewati banyak titik", "Tampilkan Semua"),
            horizontal=True,
            key="analysis_choice"
        )
        
        if len(st.session_state.data) >= 2:
            plot_fig = create_abrasion_plot(
                st.session_state.data['x_values'], 
                st.session_state.data['y_values'], 
                results, 
                analysis_choice
            )
            st.plotly_chart(plot_fig, use_container_width=True)
        else:
            st.warning("Tidak ada data yang cukup untuk membuat grafik.")

    with tab2:
        st.markdown("<h3>Nilai Tegangan Tarik (N) pada X = 50 Cycles</h3>", unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        val_original = results.get('y_at_x_50_original_curve')
        val_10_20 = results.get('y_at_x_50_pt10_20_line')
        val_ransac = results.get('y_at_x_50_ransac_line')

        with col_res1:
            st.markdown(f"""
            <div class="dark-card result-card">
                <h3>Kurva Data Asli</h3>
                <p style="font-size: 28px; font-weight: 700; color: #6B4226 !important;">
                    {f'{val_original:.2f}' if not np.isnan(val_original) else 'N/A'} N
                </p>
                <p>Menggunakan interpolasi linier pada kurva data.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"""
            <div class="dark-card result-card">
                <h3>Garis Titik 10 & 20</h3>
                <p style="font-size: 28px; font-weight: 700; color: #007BFF !important;">
                    {f'{val_10_20:.2f}' if not np.isnan(val_10_20) else 'N/A'} N
                </p>
                <p>Menggunakan garis yang ditarik antara titik data ke-10 dan ke-20.</p>
            </div>
            """, unsafe_allow_html=True)

        with col_res3:
            st.markdown(f"""
            <div class="dark-card result-card">
                <h3>Regresi RANSAC</h3>
                <p style="font-size: 28px; font-weight: 700; color: #28A745 !important;">
                    {f'{val_ransac:.2f}' if not np.isnan(val_ransac) else 'N/A'} N
                </p>
                <p>Menggunakan regresi linier robust (RANSAC) pada semua titik data.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---", unsafe_allow_html=True)
        st.markdown("<h3>Detail Titik yang Digunakan (Garis 10 & 20)</h3>", unsafe_allow_html=True)
        
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            x1, y1 = results.get('specific_x1_pt10_20'), results.get('specific_y1_pt10_20')
            st.info(f"**Titik 1 (Data ke-10):** X = {f'{x1:.2f}' if not np.isnan(x1) else 'N/A'}, Y = {f'{y1:.0f}' if not np.isnan(y1) else 'N/A'} N")
        with col_detail2:
            x2, y2 = results.get('specific_x2_pt10_20'), results.get('specific_y2_pt10_20')
            st.info(f"**Titik 2 (Data ke-20):** X = {f'{x2:.2f}' if not np.isnan(x2) else 'N/A'}, Y = {f'{y2:.0f}' if not np.isnan(y2) else 'N/A'} N")
        
# --- Jalankan Aplikasi Utama ---
main_app()

# --- Footer Aplikasi ---
st.markdown("""
<div class="radix-footer">
    <p>Dikembangkan oleh Radix | © 2024</p>
</div>
""", unsafe_allow_html=True)
