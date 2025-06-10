import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
from sklearn.linear_model import RANSACRegressor, LinearRegression
import io

# --- Konfigurasi Halaman (Paling Awal) ---
st.set_page_config(
    page_title="Radix Thread Abrasion Graph",
    page_icon="📊",
    layout="wide"
)

# --- CSS Kustom untuk Tampilan Dark Mode yang Elegan ---
st.markdown("""
<style>
    /* Main Background & Structure */
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        letter-spacing: 0.5px;
    }
    h1 {
        font-weight: 700;
        font-size: 36px;
        padding-bottom: 10px;
        border-bottom: 2px solid #4F8EF7;
    }
    h2, h3 {
        font-weight: 600;
    }
    p, li, span {
        color: #E0E0E0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4F8EF7;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #1E90FF;
        box-shadow: 0 5px 15px rgba(79, 142, 247, 0.2);
        transform: translateY(-2px);
    }
    
    /* Containers */
    .css-1v3fvcr { /* This might be an outdated class, check Streamlit's latest class for main container */
        background-color: #121212;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 8px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 20px;
        border-radius: 8px;
        background-color: #1E1E1E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4F8EF7;
    }
    
    /* DataFrame */
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        background-color: #1E1E1E;
    }
    .stDataFrame [data-testid="stTable"] {
        border: none;
    }
    .stDataFrame th {
        background-color: #2E2E2E !important;
        color: #4F8EF7 !important;
        font-weight: 600;
    }
    .stDataFrame td {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border-bottom: 1px solid #2E2E2E !important;
    }
    
    /* Footer & Hidden Elements */
    footer {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
        background-color: #1E1E1E;
    }
    div[data-baseweb="notification"] {
        background-color: #1E1E1E !important;
    }
    
    /* Additional Styling */
    .css-1r6slb0 { /* Target the main block container for expander, might need update */
        background-color: #1E1E1E !important;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        padding: 20px;
    }
     .st-emotion-cache-1wv8c4k { /* This class seems to target the expander header and content */
        background-color: #1E1E1E !important;
        border-radius: 8px;
        box_shadow: 0 2px 4px rgba(0,0,0,0.2);
        padding: 10px;
    }
    .st-emotion-cache-1wv8c4k div[data-testid="stExpanderToggle"] {
        background-color: #1E1E1E; /* Ensure expander header is dark */
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #121212;
    }
    ::-webkit-scrollbar-thumb {
        background: #2E2E2E;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4F8EF7;
    }
    
    /* Success/Error Messages */
    .success {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border-left: 4px solid #28A745 !important;
    }
    .error {
        background-color: rgba(220, 53, 69, 0.1) !important;
        border-left: 4px solid #DC3545 !important;
    }
    
    /* Custom Cards */
    .dark-card {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }

    /* Radix Header/Footer */
    .radix-logo {
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        font-size: 38px;
        color: #4F8EF7;
        margin-bottom: 5px;
        letter-spacing: 2px;
        text-shadow: 0 2px 10px rgba(79, 142, 247, 0.3);
    }
    .app-header {
        background-color: #1E1E1E;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 25px;
        display: flex;
        flex-direction: column;
        align-items: center;
        backdrop-filter: blur(5px);
    }
    .radix-footer {
        text-align: center;
        margin-top: 40px;
        padding: 15px;
        font-size: 14px;
        font-family: 'Helvetica Neue', sans-serif;
        color: #A0A0A0;
        border-top: 1px solid #2E2E2E;
        background-color: #1E1E1E;
        border-radius: 0 0 8px 8px;
    }

    /* Info Alert */
    .stInfo {
        background-color: rgba(79, 142, 247, 0.1);
        border-radius: 8px;
        padding: 15px;
        color: #FFFFFF;
        border-left: 4px solid #4F8EF7;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        font-weight: 500;
    }
    
    /* Data Editor specific styling */
    [data-testid="stDataEditor"] {
        background-color: #1E1E1E !important;
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stDataEditor"] .cell {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
    }
    [data-testid="stDataEditor"] .cell:focus {
        background-color: #2E2E2E !important;
        border: 1px solid #4F8EF7 !important;
    }
    [data-testid="stDataEditor"] .header {
        background-color: #2E2E2E !important;
        color: #4F8EF7 !important;
    }
    
    /* File Uploader specific styling */
    [data-testid="stFileUploader"] {
        background-color: #1E1E1E !important;
        border-radius: 8px;
        padding: 15px !important;
        border: 1px dashed #4F8EF7 !important;
    }
    [data-testid="stFileUploader"] p {
        color: #A0A0A0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2E2E2E !important;
        margin: 25px 0 !important;
    }
    
    /* Selection Color */
    ::selection {
        background-color: rgba(79, 142, 247, 0.3);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent {
        background-color: #1E1E1E !important;
        border-radius: 0 0 8px 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Implementasi Kode Akses ---
ACCESS_CODE = "RADIX2025" 

def check_password():
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        st.subheader("Masukkan Kode Akses")
        password_input = st.text_input("Kode Akses", type="password", key="password_input")
        if st.button("Masuk", key="login_button"):
            if password_input == ACCESS_CODE:
                st.session_state.password_entered = True
                st.rerun() # Menggunakan st.rerun() yang lebih baru
            else:
                st.error("Kode akses salah. Silakan coba lagi.")
        return False
    return True

# Panggil fungsi pengecekan password di awal aplikasi
if not check_password():
    st.stop() # Hentikan eksekusi jika password tidak benar

# --- Header Aplikasi ---
st.markdown("""
<div class="app-header">
    <div class="radix-logo">RADIX</div>
    <h1 style="margin-top: 0; color: #FFFFFF; font-size: 36px;">Thread Abrasion Analysis</h1>
    <p style="color: #A0A0A0; font-size: 16px;">A professional tool for visualizing thread abrasion data and calculating intersection values</p>
</div>
""", unsafe_allow_html=True)

# --- Data Awal ---
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# Inisialisasi session state untuk data
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)
if 'update_graph' not in st.session_state:
    st.session_state.update_graph = False
if 'p10_20_line_calculated' not in st.session_state:
    st.session_state.p10_20_line_calculated = {}
if 'ransac_line_calculated' not in st.session_state:
    st.session_state.ransac_line_calculated = {}

# --- Fungsi untuk Menghitung Garis dan Titik ---
@st.cache_data
def calculate_lines_and_points(x_values, y_values):
    results = {
        'y_at_x_50_original_curve': np.nan,
        'specific_x1_pt10_20': np.nan,
        'specific_y1_pt10_20': np.nan,
        'specific_x2_pt10_20': np.nan,
        'specific_y2_pt10_20': np.nan,
        'y_at_x_50_pt10_20_line': np.nan,
        'pt10_20_line_x_range': np.array([]),
        'pt10_20_line_y': np.array([]),
        'y_at_x_50_ransac_line': np.nan,
        'ransac_line_x': np.array([]),
        'ransac_line_y': np.array([])
    }

    if len(x_values) < 2 or len(y_values) < 2:
        return results

    # Original curve interpolation
    f = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    results['y_at_x_50_original_curve'] = float(f(50))

    # Calculate values for "Garis Antara Titik 10 & 20"
    if len(x_values) >= 20:
        results['specific_x1_pt10_20'] = x_values.iloc[9]  # 10th element (index 9)
        results['specific_y1_pt10_20'] = y_values.iloc[9]
        results['specific_x2_pt10_20'] = x_values.iloc[19] # 20th element (index 19)
        results['specific_y2_pt10_20'] = y_values.iloc[19]
    elif len(x_values) >= 2: # Fallback for datasets smaller than 20
        results['specific_x1_pt10_20'] = x_values.iloc[0]
        results['specific_y1_pt10_20'] = y_values.iloc[0]
        results['specific_x2_pt10_20'] = x_values.iloc[-1]
        results['specific_y2_pt10_20'] = y_values.iloc[-1]
    
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']) and results['specific_x1_pt10_20'] != results['specific_x2_pt10_20']:
        slope_pt10_20 = (results['specific_y2_pt10_20'] - results['specific_y1_pt10_20']) / (results['specific_x2_pt10_20'] - results['specific_x1_pt10_20'])
        intercept_pt10_20 = results['specific_y1_pt10_20'] - slope_pt10_20 * results['specific_x1_pt10_20']
        results['y_at_x_50_pt10_20_line'] = slope_pt10_20 * 50 + intercept_pt10_20

        x_min_plot = x_values.min() if not x_values.empty else 0
        x_max_plot = x_values.max() if not x_values.empty else 100
        results['pt10_20_line_x_range'] = np.linspace(min(x_min_plot, 50), max(x_max_plot, 50), 100)
        results['pt10_20_line_y'] = slope_pt10_20 * results['pt10_20_line_x_range'] + intercept_pt10_20


    # Calculate values for "Regresi Linear Robust (RANSAC)"
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
            st.warning(f"Tidak dapat melakukan Regresi RANSAC: {e}. Pastikan ada cukup variasi di data.")
            results['ransac_line_x'] = np.array([])
            results['ransac_line_y'] = np.array([])

    return results

# --- Bagian Input Data ---
tabs = st.tabs(["Manual Input", "Import from Excel"])

with tabs[0]:
    st.subheader("Manual Data Input")
    st.write("Modify y-axis values (N atau nilai benang putus):")
    
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
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Apply Changes", key="apply_changes"):
            try:
                st.session_state.data['y_values'] = edited_df['y_value'].astype(float).tolist()
                st.session_state.update_graph = True # Trigger graph update
                # Clear cached calculations
                calculate_lines_and_points.clear() 
                st.success("Data updated successfully!")
            except ValueError:
                st.error("Pastikan semua nilai Y adalah angka yang valid.")
    
    with col2:
        if st.button("Reset to Initial Values", key="reset_values"):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.update_graph = True # Trigger graph update
            # Clear cached calculations
            calculate_lines_and_points.clear()
            st.success("Data reset to initial values!")

with tabs[1]:
    st.subheader("Import Data from Excel")
    st.write("Upload an Excel file with x and y values (must have 'x_values' and 'y_values' columns)")
    
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            if 'x_values' in df.columns and 'y_values' in df.columns:
                st.write("Data Preview:")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Use This Data", key="use_imported"):
                    st.session_state.data['x_values'] = df['x_values'].astype(float).dropna()
                    st.session_state.data['y_values'] = df['y_values'].astype(float).dropna()
                    if len(st.session_state.data['x_values']) != len(st.session_state.data['y_values']):
                        st.error("Jumlah nilai X dan Y harus sama. Mohon periksa file Excel Anda.")
                    elif st.session_state.data['x_values'].empty:
                        st.warning("File Excel tidak memiliki data yang valid. Harap periksa formatnya.")
                    else:
                        st.session_state.update_graph = True # Trigger graph update
                        # Clear cached calculations
                        calculate_lines_and_points.clear()
                        st.success("Imported data applied successfully!")
            else:
                st.error("The Excel file must contain 'x_values' and 'y_values' columns.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file Excel: {e}")

# Download Sample Excel Template
st.divider()
if st.button("Download Sample Excel Template"):
    sample_df = pd.DataFrame(INITIAL_DATA)
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Click to Download",
        data=buffer,
        file_name="thread_abrasion_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --- Pilihan Metode Analisis Garis ---
st.divider()
st.subheader("Pilihan Garis dan Hasil")
display_option = st.radio(
    "Pilih garis yang ingin ditampilkan:",
    ("Hanya Kurva Asli", "Garis Antara Titik 10 & 20", "Regresi Linear Robust (RANSAC)", "Tampilkan Kedua Garis"),
    key="line_display_option"
)

# --- Grafik ---
st.subheader("Thread Abrasion Graph")

# Lakukan perhitungan hanya jika data berubah atau belum dihitung
if st.session_state.update_graph or 'calculated_results' not in st.session_state:
    x_values = pd.Series(st.session_state.data['x_values'])
    y_values = pd.Series(st.session_state.data['y_values'])
    
    if len(x_values) < 2 or len(y_values) < 2:
        st.warning("Tidak cukup data untuk menggambar grafik. Harap masukkan setidaknya 2 pasangan X dan Y.")
        st.stop() # Hentikan rendering jika data tidak cukup
    
    st.session_state.calculated_results = calculate_lines_and_points(x_values, y_values)
    st.session_state.update_graph = False # Reset flag

results = st.session_state.calculated_results
x_values = pd.Series(st.session_state.data['x_values'])
y_values = pd.Series(st.session_state.data['y_values'])

# Buat Figure Plotly
fig = go.Figure()

# Tambahkan Kurva Data Abrasi (selalu ditampilkan)
fig.add_trace(go.Scatter(
    x=x_values, 
    y=y_values,
    mode='lines+markers',
    name='Data Abrasi',
    line=dict(color='#4F8EF7', width=3),
    marker=dict(size=8, color='#4F8EF7')
))

# Tambahkan Garis Vertikal di x=50 (selalu ditampilkan)
fig.add_shape(
    type="line",
    x0=50,
    y0=y_values.min() * 0.9 if y_values.min() < 0 else 0,
    x1=50,
    y1=y_values.max() * 1.1,
    line=dict(color="#E6341E", width=2, dash="dash"),
)
fig.add_annotation(
    x=50,
    y=y_values.max() * 1.05,
    text="x=50",
    showarrow=False,
    font=dict(color="#E6341E", size=14, family="Arial, sans-serif", weight="bold")
)

# Tambahkan titik perpotongan kurva asli dengan x=50 (selalu ditampilkan)
if not np.isnan(results['y_at_x_50_original_curve']):
    fig.add_trace(go.Scatter(
        x=[50],
        y=[results['y_at_x_50_original_curve']],
        mode='markers',
        name=f'Int. Kurva Asli di x=50, y={results["y_at_x_50_original_curve"]:.2f}',
        marker=dict(size=14, color='#E6341E', symbol='circle', line=dict(width=2, color='white'))
    ))


# Render Garis dan Titik berdasarkan Pilihan Pengguna
if display_option == "Garis Antara Titik 10 & 20" or display_option == "Tampilkan Kedua Garis":
    if not np.isnan(results['specific_x1_pt10_20']) and not np.isnan(results['specific_x2_pt10_20']):
        fig.add_trace(go.Scatter(
            x=[results['specific_x1_pt10_20'], results['specific_x2_pt10_20']],
            y=[results['specific_y1_pt10_20'], results['specific_y2_pt10_20']],
            mode='markers',
            name='Titik Referensi (10 & 20)',
            marker=dict(size=12, color='#ffd700', symbol='star', line=dict(width=2, color='white'))
        ))
        fig.add_trace(go.Scatter(
            x=results['pt10_20_line_x_range'],
            y=results['pt10_20_line_y'],
            mode='lines',
            name='Garis Antara Titik 10 & 20',
            line=dict(color="#FF5733", width=3, dash="dot"),
            showlegend=True
        ))
        if not np.isnan(results['y_at_x_50_pt10_20_line']):
            fig.add_trace(go.Scatter(
                x=[50],
                y=[results['y_at_x_50_pt10_20_line']],
                mode='markers',
                name=f'Int. Garis 10-20 di x=50, y={results["y_at_x_50_pt10_20_line"]:.2f}',
                marker=dict(size=14, color='#FF5733', symbol='circle-open', line=dict(width=3, color='#FF5733'))
            ))
            y_pos_pt10_20_label = results['y_at_x_50_pt10_20_line'] + (max(y_values) * 0.05 if max(y_values) > 0 else 50)
            fig.add_annotation(
                x=50, y=y_pos_pt10_20_label, text=f"Garis 10-20: {results['y_at_x_50_pt10_20_line']:.2f}",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#FF5733',
                font=dict(size=14, color='#FF5733', family="Arial, sans-serif"),
                bordercolor="#FF5733", borderwidth=1, borderpad=4, bgcolor="rgba(30,30,30,0.7)", opacity=0.9
            )
    else:
        st.warning("Tidak cukup data untuk menggambar Garis Antara Titik 10 & 20 (membutuhkan setidaknya 20 titik data).")

if display_option == "Regresi Linear Robust (RANSAC)" or display_option == "Tampilkan Kedua Garis":
    if not np.isnan(results['y_at_x_50_ransac_line']) and len(results['ransac_line_x']) > 0:
        fig.add_trace(go.Scatter(
            x=results['ransac_line_x'],
            y=results['ransac_line_y'],
            mode='lines',
            name='Regresi Linear Robust (RANSAC)',
            line=dict(color='#00FFFF', width=3, dash='dash'),
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=[50],
            y=[results['y_at_x_50_ransac_line']],
            mode='markers',
            name=f'Int. RANSAC di x=50, y={results["y_at_x_50_ransac_line"]:.2f}',
            marker=dict(size=14, color='#00FFFF', symbol='diamond-open', line=dict(width=3, color='#00FFFF'))
        ))
        y_pos_ransac_label = results['y_at_x_50_ransac_line'] - (max(y_values) * 0.05 if results['y_at_x_50_ransac_line'] > 0 else 50)
        fig.add_annotation(
            x=50, y=y_pos_ransac_label, text=f"RANSAC: {results['y_at_x_50_ransac_line']:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='#00FFFF',
            font=dict(size=14, color='#00FFFF', family="Arial, sans-serif"),
            bordercolor="#00FFFF", borderwidth=1, borderpad=4, bgcolor="rgba(30,30,30,0.7)", opacity=0.9
        )
    else:
        st.warning("Tidak cukup data untuk menggambar Regresi Linear Robust (RANSAC).")

# Update layout (tetap sama)
fig.update_layout(
    title=None,
    xaxis_title=dict(text="Nilai yang sudah tetap", font=dict(family="Arial, sans-serif", size=14, color="#E0E0E0")),
    yaxis_title=dict(text="N atau nilai benang putus", font=dict(family="Arial, sans-serif", size=14, color="#E0E0E0")),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#E0E0E0"),
        bgcolor="rgba(30,30,30,0.7)", bordercolor="#2E2E2E", borderwidth=1
    ),
    margin=dict(l=10, r=10, t=50, b=10),
    height=600,
    template="plotly_dark",
    plot_bgcolor="#1E1E1E",
    paper_bgcolor="#1E1E1E",
    font=dict(family="Arial, sans-serif", size=12, color="#E0E0E0")
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2E2E2E', zeroline=True, zerolinewidth=1.5, zerolinecolor='#2E2E2E', tickfont=dict(color="#A0A0A0"))
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2E2E2E', zeroline=True, zerolinewidth=1.5, zerolinecolor='#2E2E2E', tickfont=dict(color="#A0A0A0"))

st.plotly_chart(fig, use_container_width=True)

# --- Tampilkan Hasil Analisis ---
st.markdown(f"""
<div class="dark-card" style="text-align: center; padding: 25px; margin-bottom: 20px;">
    <h2 style="color: #FFFFFF; margin-bottom: 5px;">Hasil Analisis pada x=50</h2>
""", unsafe_allow_html=True)

# Conditional display of results based on chosen method
if display_option == "Hanya Kurva Asli":
    final_y_value = results['y_at_x_50_original_curve']
    description_text = "Nilai perpotongan kurva data abrasi asli pada x=50."
    st.markdown(f"""
    <p style="color: #A0A0A0; font-size: 16px;">Hasil Kurva Asli:</p>
    <h1 style="color: #E6341E; font-size: 48px; margin: 10px 0;">{final_y_value:.2f}</h1>
    <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
        {description_text}
    </div>
    """, unsafe_allow_html=True)

elif display_option == "Garis Antara Titik 10 & 20":
    final_y_value = results['y_at_x_50_pt10_20_line']
    description_text = f"Nilai perpotongan garis linear antara titik ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f}) pada x=50."
    
    if not np.isnan(final_y_value):
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Garis Antara Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">{final_y_value:.2f}</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
            {description_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Garis Antara Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
            Tidak cukup data untuk menghitung garis ini (membutuhkan setidaknya 20 titik data).
        </div>
        """, unsafe_allow_html=True)

elif display_option == "Regresi Linear Robust (RANSAC)":
    final_y_value = results['y_at_x_50_ransac_line']
    description_text = "Nilai perpotongan model Regresi Linear Robust (RANSAC) pada x=50, yang mempertimbangkan outlier."
    
    if not np.isnan(final_y_value):
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Regresi Linear Robust (RANSAC):</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">{final_y_value:.2f}</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
            {description_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Regresi Linear Robust (RANSAC):</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">N/A</h1>
        <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
            Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.
        </div>
        """, unsafe_allow_html=True)

elif display_option == "Tampilkan Kedua Garis":
    st.markdown(f"""
    <p style="color: #A0A0A0; font-size: 16px;">Hasil Kurva Asli:</p>
    <h1 style="color: #E6341E; font-size: 48px; margin: 10px 0;">{results['y_at_x_50_original_curve']:.2f}</h1>
    <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Nilai perpotongan kurva data abrasi asli pada x=50.</p>
    """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_pt10_20_line']):
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Garis Antara Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">{results['y_at_x_50_pt10_20_line']:.2f}</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Berdasarkan garis linear antara titik ke-10 ({results['specific_x1_pt10_20']:.2f}, {results['specific_y1_pt10_20']:.2f}) dan titik ke-20 ({results['specific_x2_pt10_20']:.2f}, {results['specific_y2_pt10_20']:.2f}) pada x=50.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Garis Antara Titik 10 & 20:</p>
        <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">N/A</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data untuk menghitung garis ini (membutuhkan setidaknya 20 titik data).</p>
        """, unsafe_allow_html=True)

    if not np.isnan(results['y_at_x_50_ransac_line']):
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Regresi Linear Robust (RANSAC):</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">{results['y_at_x_50_ransac_line']:.2f}</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Berdasarkan model Regresi Linear Robust (RANSAC) yang mempertimbangkan outlier.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <p style="color: #A0A0A0; font-size: 16px;">Hasil Regresi Linear Robust (RANSAC):</p>
        <h1 style="color: #00FFFF; font-size: 48px; margin: 10px 0;">N/A</h1>
        <p style="color: #A0A0A0; font-size: 14px; margin-bottom: 20px;">Tidak cukup data atau data tidak valid untuk menghitung Regresi RANSAC.</p>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Tampilan Tabel Data ---
with st.expander("View Complete Data Table"):
    st.dataframe(
        pd.DataFrame({
            'x_values (Nilai yang sudah tetap)': st.session_state.data['x_values'],
            'y_values (N atau nilai benang putus)': st.session_state.data['y_values']
        }),
        hide_index=True,
        use_container_width=True
    )

# --- Informasi Essential & Footer ---
st.markdown("""
<div class="dark-card">
    <h3 style="color: #FFFFFF;">Thread Abrasion Analysis</h3>
    <p style="color: #E0E0E0;">Grafik menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y).</p>
    <ul style="color: #E0E0E0;">
        <li>The <strong style="color: #4F8EF7;">blue line</strong> adalah kurva data abrasi benang.</li>
        <li>The <strong style="color: #ffd700;">gold stars</strong> menandai titik 10 dan 20 dari dataset.</li>
        <li>The <strong style="color: #FF5733;">orange/red dashed line</strong> adalah garis linear antara titik ke-10 dan ke-20 yang diekstrapolasi ke x=50.</li>
        <li>The <strong style="color: #FF5733;">hollow circle (orange/red)</strong> menunjukkan perpotongan garis 10-20 dengan x=50.</li>
        <li>The <strong style="color: #00FFFF;">cyan dashed line</strong> adalah garis Regresi Linear Robust (RANSAC).</li>
        <li>The <strong style="color: #00FFFF;">hollow diamond (cyan)</strong> menunjukkan perpotongan garis RANSAC dengan x=50.</li>
        <li>The <strong style="color: #E6341E;">solid circle (red)</strong> menunjukkan perpotongan kurva data asli pada x=50.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

with st.expander("Informasi Tambahan & Tips Interaksi"):
    st.markdown("""
    <div style="padding: 10px;">
        <h4 style="color: #FFFFFF;">Key Points:</h4>
        <ul style="color: #E0E0E0;">
            <li>Anda dapat memilih metode analisis garis yang berbeda untuk melihat proyeksi nilai benang putus pada x=50.</li>
            <li>Garis <strong style="color: #FF5733;">Antara Titik 10 & 20</strong> adalah proyeksi linear sederhana dari dua titik data spesifik.</li>
            <li>Garis <strong style="color: #00FFFF;">Regresi Linear Robust (RANSAC)</strong> mencoba menemukan garis terbaik yang sesuai dengan mayoritas titik data sambil mengabaikan outlier.</li>
            <li>Nilai pada garis yang dipilih saat x=50 ditampilkan sebagai "hasil" utama untuk analisis ini.</li>
        </ul>
        
        <h4 style="color: #FFFFFF; margin-top: 20px;">Tips Interaksi:</h4>
        <ul style="color: #E0E0E0;">
            <li>Arahkan kursor ke titik untuk melihat nilainya.</li>
            <li>Klik dan seret untuk memperbesar area tertentu.</li>
            <li>Klik dua kali untuk mengatur ulang tampilan.</li>
            <li>Gunakan toolbar di kanan atas grafik untuk opsi lainnya.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="radix-footer">
    <p>Developed by <span style="font-weight: bold; color: #4F8EF7;">RADIX</span> &copy; 2025</p>
    <p>Professional Thread Abrasion Analysis Solutions</p>
</div>
""", unsafe_allow_html=True)
