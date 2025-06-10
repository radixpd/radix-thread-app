import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import interpolate
# from utils import interpolate_value_at_x # Jika utils.py ada, pastikan juga di repo GitHub Anda
import io

# --- MULAI IMPLEMENTASI KODE AKSES ---
# Tentukan kode rahasia Anda
ACCESS_CODE = "RADIX2025" # Anda bisa mengubah ini ke kode apa pun yang Anda inginkan

def check_password():
    """Mengembalikan True jika pengguna memasukkan kata sandi yang benar, False jika tidak."""
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        st.subheader("Masukkan Kode Akses")
        password_input = st.text_input("Kode Akses", type="password", key="password_input")
        if st.button("Masuk", key="login_button"):
            if password_input == ACCESS_CODE:
                st.session_state.password_entered = True
                st.experimental_rerun() # Jalankan ulang untuk menyembunyikan layar login
            else:
                st.error("Kode akses salah. Silakan coba lagi.")
        return False
    return True

# Panggil fungsi pengecekan kata sandi di awal aplikasi Anda
if not check_password():
    st.stop() # Hentikan eksekusi jika kata sandi tidak dimasukkan atau salah
# --- AKHIR IMPLEMENTASI KODE AKSES ---


# Set konfigurasi halaman
st.set_page_config(
    page_title="Radix Thread Abrasion Graph",
    page_icon="📊",
    layout="wide"
)

# Custom CSS untuk gaya Dark Mode yang elegan
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
    .css-1v3fvcr {
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
    .css-1r6slb0 {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        padding: 20px;
        background-color: #1E1E1E !important;
    }
    
    /* Custom Classes */
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
    .stInfo {
        background-color: rgba(79, 142, 247, 0.1);
        border-radius: 8px;
        padding: 15px;
        color: #FFFFFF;
        border-left: 4px solid #4F8EF7;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        font-weight: 500;
    }
    
    /* Data Editor */
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
    
    /* File Uploader */
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
</style>
""", unsafe_allow_html=True)

# Header Aplikasi dengan Logo Radix
st.markdown("""
<div class="app-header">
    <div class="radix-logo">RADIX</div>
    <h1 style="margin-top: 0; color: #FFFFFF; font-size: 36px;">Analisis Abrasi Benang</h1>
    <p style="color: #A0A0A0; font-size: 16px;">Alat profesional untuk memvisualisasikan data abrasi benang dan menghitung nilai persimpangan</p>
</div>
""", unsafe_allow_html=True)

# Tentukan data awal
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# Inisialisasi session state untuk data jika belum ada
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)

if 'update_graph' not in st.session_state:
    st.session_state.update_graph = False

# Buat tab untuk berbagai cara memasukkan data
tabs = st.tabs(["Input Manual", "Impor dari Excel"])

with tabs[0]:
    st.subheader("Input Data Manual")
    st.write("Modifikasi nilai sumbu-y (N atau nilai benang putus):")
    
    # Buat dataframe yang dapat diedit untuk nilai-y
    edited_data = pd.DataFrame({
        'x_value': st.session_state.data['x_values'],
        'y_value': st.session_state.data['y_values']
    })
    
    # Buat dapat diedit
    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=True,
        use_container_width=True,
        key="data_editor",
        on_change=None # Jangan update otomatis
    )
    
    # Tambahkan tombol untuk menerapkan perubahan
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Terapkan Perubahan", key="apply_changes"):
            st.session_state.data['y_values'] = edited_df['y_value'].tolist()
            st.session_state.update_graph = True
            st.success("Data berhasil diperbarui!")
    
    with col2:
        # Tombol untuk mengatur ulang ke nilai awal
        if st.button("Reset ke Nilai Awal", key="reset_values"):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.session_state.update_graph = True
            st.success("Data direset ke nilai awal!")

with tabs[1]:
    st.subheader("Impor Data dari Excel")
    st.write("Unggah file Excel dengan nilai x dan y (harus memiliki kolom 'x_values' dan 'y_values')")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Baca file Excel
            df = pd.read_excel(uploaded_file)
            
            # Periksa apakah kolom yang diperlukan ada
            if 'x_values' in df.columns and 'y_values' in df.columns:
                # Pratinjau data
                st.write("Pratinjau Data:")
                st.dataframe(df.head(), use_container_width=True)
                
                # Tombol untuk menerapkan data yang diimpor
                if st.button("Gunakan Data Ini", key="use_imported"):
                    st.session_state.data = df[['x_values', 'y_values']]
                    st.session_state.update_graph = True
                    st.success("Data yang diimpor berhasil diterapkan!")
            else:
                st.error("File Excel harus berisi kolom 'x_values' dan 'y_values'.")
        except Exception as e:
            st.error(f"Error membaca file Excel: {e}")

# Tampilkan template Excel contoh untuk diunduh
st.divider()
if st.button("Unduh Template Excel Contoh"):
    # Buat DataFrame contoh
    sample_df = pd.DataFrame(INITIAL_DATA)
    
    # Buat buffer untuk menyimpan file Excel
    buffer = io.BytesIO()
    sample_df.to_excel(buffer, index=False)
    buffer.seek(0)
    
    # Buat tombol unduh
    st.download_button(
        label="Klik untuk Mengunduh",
        data=buffer,
        file_name="thread_abrasion_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Buat bagian grafik
st.divider()
st.subheader("Grafik Abrasi Benang")

# Plot grafik
if st.session_state.update_graph or not 'fig' in st.session_state:
    # Dapatkan data dari session state
    x_values = st.session_state.data['x_values']
    y_values = st.session_state.data['y_values']
    
    # Buat fungsi interpolasi
    f = interpolate.interp1d(x_values, y_values, kind='linear', fill_value='extrapolate')
    
    # Temukan indeks untuk titik 10 dan 20 dalam dataset
    # Dapatkan indeks titik-titik terdekat dengan posisi ke-10 dan ke-20
    x_values_list = list(x_values)
    
    # Temukan titik 10 (kira-kira nilai ke-10 dalam dataset)
    point_10_index = 9 # Indeks berbasis 0, jadi elemen ke-10 ada di indeks 9
    if point_10_index < len(x_values_list):
        specific_x1 = x_values_list[point_10_index]
        specific_y1 = y_values[point_10_index]
    else:
        # Fallback jika data lebih pendek dari yang diharapkan
        specific_x1 = x_values_list[len(x_values_list) // 3] # Kira-kira 1/3 dari keseluruhan
        specific_y1 = y_values[len(x_values_list) // 3]
    
    # Temukan titik 20 (kira-kira nilai ke-20 dalam dataset)
    point_20_index = 19 # Indeks berbasis 0, jadi elemen ke-20 ada di indeks 19
    if point_20_index < len(x_values_list):
        specific_x2 = x_values_list[point_20_index]
        specific_y2 = y_values[point_20_index]
    else:
        # Fallback jika data lebih pendek dari yang diharapkan
        specific_x2 = x_values_list[2 * len(x_values_list) // 3] # Kira-kira 2/3 dari keseluruhan
        specific_y2 = y_values[2 * len(x_values_list) // 3]
    
    # Hitung kemiringan dan intersep garis yang menghubungkan kedua titik ini
    slope = (specific_y2 - specific_y1) / (specific_x2 - specific_x1)
    intercept = specific_y1 - slope * specific_x1
    
    # Fungsi untuk menghitung nilai y menggunakan persamaan garis
    def line_equation(x):
        return slope * x + intercept
    
    # Hitung nilai y pada x=50 menggunakan persamaan garis
    y_at_x_50_line = line_equation(50)
    
    # Interpolasi kurva asli pada titik-titik khusus
    y_at_x_10 = float(f(10))
    y_at_x_20 = float(f(20))
    y_at_x_50 = float(f(50))
    
    # Buat gambar
    fig = go.Figure()
    
    # Tambahkan garis utama
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values,
        mode='lines+markers',
        name='Data Abrasi',
        line=dict(color='#4F8EF7', width=3),
        marker=dict(size=8, color='#4F8EF7')
    ))
    
    # Tambahkan titik-titik khusus (Titik 10 dan Titik 20)
    fig.add_trace(go.Scatter(
        x=[specific_x1, specific_x2],
        y=[specific_y1, specific_y2],
        mode='markers',
        name='Titik Referensi Terpilih',
        marker=dict(size=12, color='#ffd700', symbol='star', line=dict(width=2, color='white'))
    ))
    
    # Tambahkan garis yang menghubungkan dua titik khusus dan memanjang ke x=50
    fig.add_shape(
        type="line",
        x0=specific_x1,
        y0=specific_y1,
        x1=specific_x2,
        y1=specific_y2,
        line=dict(color="#FF5733", width=3),
    )
    
    # Perpanjang garis ke x=50
    fig.add_shape(
        type="line",
        x0=specific_x2,
        y0=specific_y2,
        x1=50,
        y1=y_at_x_50_line,
        line=dict(color="#FF5733", width=3, dash="dot"),
    )
    
    # Tambahkan titik pada persimpangan garis dengan x=50
    fig.add_trace(go.Scatter(
        x=[50],
        y=[y_at_x_50_line],
        mode='markers',
        name=f'Persimpangan Garis pada x=50, y={y_at_x_50_line:.2f}',
        marker=dict(size=14, color='#FF5733', symbol='circle-open', line=dict(width=3, color='#FF5733'))
    ))
    
    # Tambahkan label untuk ekstrapolasi linear pada x=50
    fig.add_annotation(
        x=50,
        y=y_at_x_50_line + 50, # Offset untuk menghindari tumpang tindih
        text=f"Nilai Garis: {y_at_x_50_line:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#FF5733',
        font=dict(size=14, color='#FF5733', family="Arial, sans-serif"),
    )
    
    # Tambahkan garis vertikal pada x=17.7 (titik 10) dan x=40.4 (titik 20)
    for x_pos, y_val, color, name, index in [
        (specific_x1, specific_y1, "#ffd700", f"Titik 10 ({specific_x1}, {specific_y1})", 10),
        (specific_x2, specific_y2, "#ffd700", f"Titik 20 ({specific_x2}, {specific_y2})", 20),
    ]:
        # Tambahkan garis putus-putus vertikal
        fig.add_shape(
            type="line",
            x0=x_pos,
            y0=0,
            x1=x_pos,
            y1=max(y_values) * 1.1,
            line=dict(color=color, width=1.5, dash="dash"),
        )
        
        # Tambahkan label teks untuk garis vertikal
        fig.add_annotation(
            x=x_pos,
            y=y_val - 50, # Offset di bawah titik
            text=f"Titik {index}",
            showarrow=False,
            font=dict(color=color, size=12, family="Arial, sans-serif", weight="bold")
        )
    
    # Tambahkan garis vertikal pada x=50
    fig.add_shape(
        type="line",
        x0=50,
        y0=0,
        x1=50,
        y1=max(y_values) * 1.1,
        line=dict(color="#E6341E", width=2, dash="dash"),
    )
    
    # Tambahkan label teks untuk garis x=50
    fig.add_annotation(
        x=50,
        y=max(y_values) * 1.05,
        text="x=50",
        showarrow=False,
        font=dict(color="#E6341E", size=14, family="Arial, sans-serif", weight="bold")
    )
    
    # Tambahkan titik pada persimpangan dengan x=50
    fig.add_trace(go.Scatter(
        x=[50],
        y=[y_at_x_50],
        mode='markers',
        name=f'Persimpangan pada x=50, y={y_at_x_50:.2f}',
        marker=dict(size=14, color='#E6341E', symbol='circle', line=dict(width=2, color='white'))
    ))
    
    # Perbarui tata letak dengan gaya yang lebih elegan
    fig.update_layout(
        title=None,
        xaxis_title=dict(text="Nilai yang sudah tetap", font=dict(family="Arial, sans-serif", size=14, color="#455d7a")),
        yaxis_title=dict(text="N atau nilai benang putus", font=dict(family="Arial, sans-serif", size=14, color="#455d7a")),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            font=dict(family="Arial, sans-serif", size=12, color="#455d7a"),
            bordercolor="#e6e6e6",
            borderwidth=1
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=600,
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="#455d7a")
    )
    
    # Tambahkan kisi untuk keterbacaan yang lebih baik
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f1f1f1',
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='#e6e6e6'
    )
    
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='#f1f1f1',
        zeroline=True,
        zerolinewidth=1.5,
        zerolinecolor='#e6e6e6'
    )
    
    # Simpan di session state
    st.session_state.fig = fig
    st.session_state.y_at_x_50 = y_at_x_50
    st.session_state.update_graph = False

# Tampilkan grafik
st.plotly_chart(st.session_state.fig, use_container_width=True)

# Hapus kolom yang tidak terpakai
# intersection_col1, intersection_col2 = st.columns(2)

# Buat fungsi untuk menghitung ulang nilai garis menggunakan data session state
def recalculate_line_values():
    # Dapatkan data saat ini
    x_values = st.session_state.data['x_values']
    y_values = st.session_state.data['y_values']
    
    # Gunakan titik ke-10 dan ke-20 dalam dataset (indeks 9 dan 19)
    x_values_list = list(x_values)
    
    point_10_index = 9 # Indeks berbasis 0, jadi elemen ke-10 ada di indeks 9
    if point_10_index < len(x_values_list):
        specific_x1 = x_values_list[point_10_index]
        specific_y1 = y_values[point_10_index]
    else:
        # Fallback jika data lebih pendek dari yang diharapkan
        specific_x1 = x_values_list[len(x_values_list) // 3]
        specific_y1 = y_values[len(x_values_list) // 3]
    
    point_20_index = 19 # Indeks berbasis 0, jadi elemen ke-20 ada di indeks 19
    if point_20_index < len(x_values_list):
        specific_x2 = x_values_list[point_20_index]
        specific_y2 = y_values[point_20_index]
    else:
        # Fallback jika data lebih pendek dari yang diharapkan
        specific_x2 = x_values_list[2 * len(x_values_list) // 3]
        specific_y2 = y_values[2 * len(x_values_list) // 3]
    
    # Hitung kemiringan dan intersep
    slope = (specific_y2 - specific_y1) / (specific_x2 - specific_x1)
    intercept = specific_y1 - slope * specific_x1
    
    # Hitung nilai pada x=50 menggunakan persamaan garis
    return slope * 50 + intercept, specific_x1, specific_y1, specific_x2, specific_y2

# Hitung persimpangan garis dan dapatkan titik-titik
y_line_intersection, p10_x, p10_y, p20_x, p20_y = recalculate_line_values()

# Tampilkan hasil dalam format yang lebih bersih
st.markdown(f"""
<div class="dark-card" style="text-align: center; padding: 25px; margin-bottom: 20px;">
    <h2 style="color: #FFFFFF; margin-bottom: 5px;">Hasil Analisis pada x=50</h2>
    <h1 style="color: #FF5733; font-size: 48px; margin: 10px 0;">{y_line_intersection:.2f}</h1>
    <p style="color: #A0A0A0; font-size: 16px;">Nilai perpotongan garis pada x=50</p>
    <div style="margin-top: 15px; font-size: 14px; color: #A0A0A0;">
        Berdasarkan garis antara titik 10 ({p10_x}, {p10_y}) dan titik 20 ({p20_x}, {p20_y})
    </div>
</div>
""", unsafe_allow_html=True)

# Tampilkan tampilan data tabel
with st.expander("Lihat Tabel Data Lengkap"):
    st.dataframe(
        pd.DataFrame({
            'x_values (Nilai yang sudah tetap)': st.session_state.data['x_values'],
            'y_values (N atau nilai benang putus)': st.session_state.data['y_values']
        }),
        hide_index=True,
        use_container_width=True
    )

# Tambahkan informasi penting tentang grafik
st.markdown("""
<div class="dark-card">
    <h3 style="color: #FFFFFF;">Analisis Abrasi Benang</h3>
    <p style="color: #E0E0E0;">Grafik menunjukkan hubungan antara nilai tetap (sumbu-x) dan nilai benang putus (sumbu-y).</p>
    <ul style="color: #E0E0E0;">
        <li><strong style="color: #4F8EF7;">Garis biru</strong> adalah kurva data abrasi benang.</li>
        <li><strong style="color: #ffd700;">Bintang emas</strong> menandai titik 10 dan 20 dari dataset.</li>
        <li><strong style="color: #FF5733;">Garis oranye/merah</strong> menghubungkan kedua titik tersebut dan diperpanjang ke x=50.</li>
        <li><strong style="color: #FF5733;">Lingkaran kosong</strong> menunjukkan perpotongan garis dengan x=50.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Tambahkan informasi rinci dalam expander
with st.expander("Informasi Tambahan & Tips Interaksi"):
    st.markdown("""
    <div style="padding: 10px;">
        <h4 style="color: #FFFFFF;">Poin Penting:</h4>
        <ul style="color: #E0E0E0;">
            <li>Garis yang ditarik antara titik 10 dan 20 memberikan proyeksi linear ke x=50</li>
            <li>Nilai pada garis saat x=50 dianggap sebagai "hasil" resmi untuk analisis ini</li>
        </ul>
        
        <h4 style="color: #FFFFFF; margin-top: 20px;">Tips Interaksi:</h4>
        <ul style="color: #E0E0E0;">
            <li>Arahkan kursor ke titik untuk melihat nilainya</li>
            <li>Klik dan seret untuk memperbesar area tertentu</li>
            <li>Klik dua kali untuk mengatur ulang tampilan</li>
            <li>Gunakan toolbar di kanan atas grafik untuk opsi lainnya</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Tambahkan footer Radix
st.markdown("""
<div class="radix-footer">
    <p>Dikembangkan oleh <span style="font-weight: bold; color: #4F8EF7;">RADIX</span> &copy; 2025</p>
    <p>Solusi Analisis Abrasi Benang Profesional</p>
</div>
""", unsafe_allow_html=True)
