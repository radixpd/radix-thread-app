import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor, LinearRegression
import io

# --- Konfigurasi Halaman (Paling Awal) ---
st.set_page_config(
    page_title="Radix Thread Abrasion",
    page_icon="🧵", # Mengganti ikon
    layout="wide"
)

# --- CSS Kustom untuk Tampilan Dark Mode Minimalis & Elegan ---
st.markdown("""
<style>
    /* General Styles */
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding-top: 20px; /* Adjust top padding for app */
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: 0.5px;
    }
    h1 {
        font-weight: 700;
        font-size: 38px;
        padding-bottom: 10px;
        border-bottom: 2px solid #4F8EF7;
        text-align: center; /* Center main title */
    }
    h2 {
        font-weight: 600;
        font-size: 28px;
        color: #4F8EF7; /* Highlight section titles */
        margin-bottom: 15px;
        border-bottom: 1px solid #2E2E2E;
        padding-bottom: 5px;
    }
    h3 {
        font-weight: 600;
        font-size: 22px;
        color: #FFFFFF;
    }
    p, li, span {
        color: #E0E0E0;
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.6;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4F8EF7;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 10px 20px; /* Larger buttons */
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1E90FF;
        box-shadow: 0 5px 15px rgba(79, 142, 247, 0.2);
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0;
        font-weight: 500;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 10px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 25px;
        border-radius: 10px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4F8EF7;
        border-radius: 5px;
    }

    /* Radio Buttons */
    .stRadio > label {
        color: #E0E0E0;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .stRadio > div {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stRadio [data-baseweb="radio"] {
        background-color: #2E2E2E;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 5px;
        transition: background-color 0.3s ease;
    }
    .stRadio [data-baseweb="radio"]:hover {
        background-color: #3A3A3A;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: #4F8EF7 !important;
        color: white;
        border: 1px solid #4F8EF7;
    }
    .stRadio [data-baseweb="radio"] span:last-child { /* text of the radio button */
        color: #E0E0E0;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] span:last-child {
        color: white;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }

    /* Radix Header */
    .app-header {
        background-color: #1E1E1E;
        padding: 30px;
        border-radius: 15px; /* More rounded */
        box-shadow: 0 6px 25px rgba(0,0,0,0.3); /* Stronger shadow */
        margin-bottom: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(5px);
        border: 1px solid #2E2E2E; /* Subtle border */
    }
    .radix-logo {
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        font-size: 42px; /* Larger logo */
        color: #4F8EF7;
        margin-bottom: 8px;
        letter-spacing: 3px;
        text-shadow: 0 3px 15px rgba(79, 142, 247, 0.4);
    }

    /* Footer */
    .radix-footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        font-size: 14px;
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0;
        border-top: 1px solid #2E2E2E;
        background-color: #1E1E1E;
        border-radius: 0 0 10px 10px; /* Match tab border radius */
    }

    /* Other elements */
    hr {
        border-color: #2E2E2E !important;
        margin: 30px 0 !important;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    
    /* Streamlit specific adjustments for better dark mode */
    [data-testid="stToolbar"] { /* Hide toolbar by default if not needed */
        visibility: hidden !important;
        height: 0px !important;
        position: fixed !important;
    }
    /* Show toolbar on hover (optional) */
    .stApp:hover [data-testid="stToolbar"] {
        visibility: visible !important;
        height: auto !important;
    }

    /* For data editor and file uploader to blend better */
    [data-testid="stDataEditor"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2E2E2E;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #4F8EF7;
        border-radius: 10px;
        padding: 20px;
        background-color: #1E1E1E;
    }
    [data-testid="stFileUploaderDropzone"] p {
        color: #A0A0A0;
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
        col_pw1, col_pw2, col_pw3 = st.columns([1,1,1])
        with col_pw2:
            if st.button("Masuk", key="login_button", use_container_width=True):
                if password_input == ACCESS_CODE:
                    st.session_state.password_entered = True
                    st.rerun()
                else:
                    st.error("Kode akses salah. Silakan coba lagi.")
        st.markdown("<br><br>", unsafe_allow_html=True) # Add some space
        return False
    return True

# Panggil fungsi pengecekan password di awal aplikasi
if not check_password():
    st.stop()

# --- Header Aplikasi ---
st.markdown("""
<div class="app-header">
    <div class="radix-logo">RADIX</div>
    <h1 style="margin-top: 0; color: #FFFFFF; font-size: 30px; border-bottom: none; padding-bottom: 0;">Analisis Abrasi Benang</h1>
    <p style="color: #A0A0A0; font-size: 16px; text-align: center;">Alat profesional untuk visualisasi data dan perhitungan nilai perpotongan</p>
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
    st.session_state.calculated_results = {} # Inisialisasi kosong

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
            st.warning(f"Error Regresi RANSAC: {e}. Pastikan data memiliki variasi.")
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])
    else:
        st.warning("Data tidak cukup untuk Regresi RANSAC (minimal 2 titik).")

    return results

# --- Bagian Input Data ---
st.subheader("Input Data")
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.write("Ubah nilai Y (N atau nilai benang putus):")
    
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
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes", use_container_width=True):
            try:
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
            df = pd.read_excel(uploaded_file)
            if 'x_values' in df.columns and 'y_values' in df.columns:
                st.write("Pratinjau Data:")
                st.dataframe(df.head(), use_container_width=True)
                if st.button("Gunakan Data Ini", key="use_imported", use_container_width=True):
                    st.session_state.data['x_values'] = df['x_values'].astype(float).dropna()
                    st.session_state.data['y_values'] = df['y_values'].astype(float).dropna()
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
        st.stop()
    
    st.session_state.calculated_results = calculate_lines_and_points(x_values, y_values)
    st.session_state.update_graph = False

results = st.session_state.calculated_results
x_values = pd.Series(st.session_state.data['x_values'])
y_values = pd.Series(st.session_state.data['y_values'])

st.subheader("Visualisasi & Hasil Analisis")

# Pilihan tampilan grafik
graph_display_option = st.radio(
    "Pilih grafik yang ingin ditampilkan:",
    ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis Regresi RANSAC", "Tampilkan Semua Garis"),
    key="graph_display_option",
    horizontal=True # Tata letak horizontal
)

# Render Grafik
st.markdown("---") # Garis pemisah untuk grafik
st.write("#### Grafik Abrasi Benang")

fig = go.Figure()

# Tambahkan Kurva Data Abrasi (selalu)
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_values,
    mode='lines+markers',
    name='Data Abrasi',
    line=dict(color='#4F8EF7', width=3),
    marker=dict(size=8, color='#4F8EF7')
))

# Tambahkan Garis Vertikal di x=50 (selalu)
fig.add_shape(
    type="line",
    x0=50, y0=y_values.min() * 0.9 if y_values.min() < 0 else 0,
    x1=50, y1=y_values.max() * 1.1,
    line=dict(color="#E6341E", width=2, dash="dash"),
)
fig.add_annotation(
    x=50, y=y_values.max() * 1.05, text="x=50", showarrow=False,
    font=dict(color="#E6341E", size=14, family="Arial, sans-serif", weight="bold")
)

# Tambahkan titik perpotongan kurva asli dengan x=50 (selalu)
if not np.isnan(results['y_at_x_50_original_curve']):
    fig.add_trace(go.Scatter(
        x=[50], y=[results['y_at_x_50_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=50, y={results["y_at_x_50_original_curve"]:.2f}',
        marker=dict(size=14, color='#E6341E', symbol='circle', line=dict(width=2, color='white'))
    ))

# Kondisional untuk Garis Titik 10 & 20
if graph_display_option in ["Garis Titik 10 & 20", "Tampilkan Semua Garis"]:
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']):
        fig.add_trace(go.Scatter(
            x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
            y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
            mode='markers', name='Titik Referensi (10 & 20)',
            marker=dict(size=12, color='#ffd700', symbol='star', line=dict(width=2, color='white'))
        ))
        fig.add_trace(go.Scatter(
            x=results['pt10_20_line_x_range'], y=results['pt10_20_line_y'],
            mode='lines', name='Garis Titik 10 & 20',
            line=dict(color="#FF5733", width=3, dash="dot"), showlegend=True
        ))
        if not np.isnan(results['y_at_x_50_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[50], y=[results['y_at_x_50_pt10_20_line']],
                mode='markers', name=f'Int. Garis 10-20 di x=50, y={results["y_at_x_50_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#FF5733', symbol='circle-open', line=dict(width=3, color='#FF5733'))
            ))
            y_pos_pt10_20_label = results['y_at_x_50_pt10_20_line'] + (y_values.max() * 0.05 if y_values.max() > 0 else 50)
            fig.add_annotation(
                x=50, y=y_pos_pt10_20_label, text=f"Garis 10-20: {results['y_at_x_50_pt10_20_line']:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#FF5733',
                font=dict(size=14, color='#FF5733', family="Arial, sans-serif"),
                bordercolor="#FF5733", borderwidth=1, borderpad=4, bgcolor="rgba(30,30,30,0.7)", opacity=0.9
            )

# Kondisional untuk Garis Regresi RANSAC
if graph_display_option in ["Garis Regresi RANSAC", "Tampilkan Semua Garis"]:
    if not np.isnan(results['y_at_x_50_ransac_line']) and len(results['ransac_line_x']) > 0:
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'], y=results['ransac_line_y'],
            mode='lines', name='Regresi RANSAC',
            line=dict(color='#00FFFF', width=3, dash='dash'), showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[50], y=[results['y_at_x_50_ransac_line']],
            mode='markers', name=f'Int. RANSAC di x=50, y={results["y_at_x_50_ransac_line"]:.2f}',
            marker=dict(size=14, color='#00FFFF', symbol='diamond-open', line=dict(width=3, color='#00FFFF'))
        ))
        y_pos_ransac_label = results['y_at_x_50_ransac_line'] - (y_values.max() * 0.05 if results['y_at_x_50_ransac_line'] > 0 else 50)
        fig.add_annotation(
            x=50, y=y_pos_ransac_label, text=f"RANSAC: {results['y_at_x_50_ransac_line']:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#00FFFF',
            font=dict(size=14, color='#00FFFF', family="Arial, sans-serif"),
            bordercolor="#00FFFF", borderwidth=1, borderpad=4, bgcolor="rgba(30,30,30,0.7)", opacity=0.9
        )

# Update layout
fig.update_layout(
    xaxis_title="Nilai Tetap (x)",
    yaxis_title="Nilai Benang Putus (N)",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=12, color="#E0E0E0"), bgcolor="rgba(30,30,30,0.7)", borderwidth=1
    ),
    margin=dict(l=10, r=10, t=50, b=10), height=550, template="plotly_dark",
    plot_bgcolor="#1E1E1E", paper_bgcolor="#1E1E1E", font=dict(color="#E0E0E0")
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2E2E2E', zeroline=True, zerolinewidth=1.5, zerolinecolor='#2E2E2E', tickfont=dict(color="#A0A0A0"))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2E2E2E', zeroline=True, zerolinewidth=1.5, zerolinecolor='#2E2E2E', tickfont=dict(color="#A0A0A0"))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---") # Garis pemisah untuk hasil

# Tampilkan Hasil Analisis
st.write("#### Hasil Nilai Perpotongan pada x=50")

# Pilihan tampilan hasil
result_display_option = st.radio(
    "Pilih hasil yang ingin ditampilkan:",
    ("Hasil Kurva Asli", "Hasil Garis Titik 10 & 20", "Hasil Garis Regresi RANSAC", "Tampilkan Semua Hasil"),
    key="result_display_option",
    horizontal=True
)

st.markdown(f"""
<div class="dark-card" style="text-align: center; padding: 25px; margin-bottom: 20px;">
""", unsafe_allow_html=True)

if result_display_option == "Hasil Kurva Asli":
    val = results['y_at_x_50_original_curve']
    desc = "Nilai perpotongan kurva data asli pada x=50."
    st.markdown(f"""
    <p style="color: #A0A0A0; font-size: 16px;">Kurva Data Asli:</p>
    <h1 style="color: #E6341E; font-size: 48px; margin: 10px 0;">{val:.2f}</h1>
    <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">{desc}</div>
    """, unsafe_allow_html=True)

elif result_display_option == "Hasil Garis Titik 10 & 20":
    val = results['y_at_x_50_pt10_20_line']
    if not np.isnan(val):
        desc = f"Berdasarkan garis linear antara titik ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f})."
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">{val:.2f}</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">{desc}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
            Tidak cukup data untuk menghitung garis ini (minimal 20 titik).
        </div>
        """, unsafe_allow_html=True)

elif result_display_option == "Hasil Garis Regresi RANSAC":
    val = results['y_at_x_50_ransac_line']
    if not np.isnan(val):
        desc = "Berdasarkan model Regresi Linear Robust (RANSAC) yang mempertimbangkan outlier."
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Regresi RANSAC:</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">{val:.2f}</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">{desc}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Regresi RANSAC:</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
            Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.
        </div>
        """, unsafe_allow_html=True)

elif result_display_option == "Tampilkan Semua Hasil":
    # Hasil Kurva Asli
    val_ori = results['y_at_x_50_original_curve']
    st.markdown(f"""
    <p style="color: #A0A0A0; font-size: 16px;">Kurva Data Asli:</p>
    <h1 style="color: #E6341E; font-size: 48px; margin: 10px 0;">{val_ori:.2f}</h1>
    <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Nilai perpotongan kurva data asli pada x=50.</p>
    """, unsafe_allow_html=True)

    # Hasil Garis Titik 10 & 20
    val_10_20 = results['y_at_x_50_pt10_20_line']
    if not np.isnan(val_10_20):
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">{val_10_20:.2f}</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Berdasarkan garis linear antara titik ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f}).</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">N/A</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data untuk menghitung garis ini (minimal 20 titik).</p>
        """, unsafe_allow_html=True)
    
    # Hasil Garis Regresi RANSAC
    val_ransac = results['y_at_x_50_ransac_line']
    if not np.isnan(val_ransac):
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Regresi RANSAC:</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">{val_ransac:.2f}</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Berdasarkan model Regresi Linear Robust (RANSAC) yang mempertimbangkan outlier.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Garis Regresi RANSAC:</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">N/A</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.</p>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Tampilan Tabel Data Lengkap ---
with st.expander("Lihat Tabel Data Lengkap"):
    st.dataframe(
        pd.DataFrame({
            'Nilai Tetap (x)': st.session_state.data['x_values'],
            'Nilai Benang Putus (N)': st.session_state.data['y_values']
        }),
        hide_index=True,
        use_container_width=True
    )

# --- Informasi & Footer ---
st.markdown("""
<div class="dark-card">
    <h3>Tentang Grafik</h3>
    <p>Grafik ini menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y). Anda dapat memilih jenis garis yang ingin ditampilkan untuk analisis.</p>
    <ul>
        <li><strong style="color: #4F8EF7;">Kurva Data Asli:</strong> Garis biru mewakili data abrasi benang yang Anda masukkan.</li>
        <li><strong style="color: #ffd700;">Titik Referensi (10 & 20):</strong> Bintang emas menandai titik data ke-10 dan ke-20.</li>
        <li><strong style="color: #FF5733;">Garis Titik 10 & 20:</strong> Garis putus-putus merah-oranye adalah proyeksi linear antara titik data ke-10 dan ke-20.</li>
        <li><strong style="color: #00FFFF;">Garis Regresi RANSAC:</strong> Garis putus-putus cyan adalah model regresi linear robust yang mengabaikan outlier data.</li>
        <li><strong style="color: #E6341E;">Garis Vertikal x=50:</strong> Garis putus-putus merah menunjukkan titik referensi x=50.</li>
    </ul>
    <h3>Tips Interaksi</h3>
    <ul>
        <li>Arahkan kursor ke titik atau garis untuk melihat detail nilainya.</li>
        <li>Klik dan seret di grafik untuk memperbesar area tertentu.</li>
        <li>Klik dua kali pada grafik untuk mengatur ulang tampilan ke awal.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="radix-footer">
    <p>Dikembangkan oleh <span style="font-weight: bold; color: #4F8EF7;">RADIX</span> &copy; 2025</p>
    <p>Solusi Analisis Abrasi Benang Profesional</p>
</div>
""", unsafe_allow_html=True)
