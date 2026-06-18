"""
Analisis Abrasi Benang - RADIX
Tampilan disederhanakan: mengandalkan tema bawaan Streamlit (lihat .streamlit/config.toml)
alih-alih CSS override yang berat, supaya konsisten dan mudah dirawat.
"""

import io
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import interpolate
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Dependensi opsional - jangan sampai bikin seluruh app crash kalau belum terinstal
try:
    from docx import Document
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import kaleido  # noqa: F401  (dibutuhkan plotly fig.write_image / to_image)
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Analisis Abrasi Benang", page_icon="🧵", layout="wide")

ACCESS_CODE = "RADIX2025"
TARGET_X_VALUE = 50

INITIAL_DATA = {
    "x_values": [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7,
                 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2,
                 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    "y_values": [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345,
                 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635,
                 667, 770, 805, 974],
}

# Sedikit polish visual saja - bukan override total seperti sebelumnya
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(INITIAL_DATA)
if "password_entered" not in st.session_state:
    st.session_state.password_entered = False


# ----------------------------------------------------------------------------
# Gerbang akses
# ----------------------------------------------------------------------------
def check_password() -> bool:
    if st.session_state.password_entered:
        return True

    st.markdown("<h3 style='text-align:center;'>🔒 Akses Aplikasi</h3>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.container(border=True):
            pw = st.text_input("Kode akses", type="password", placeholder="Masukkan kode akses")
            if st.button("Masuk", use_container_width=True, type="primary"):
                if pw == ACCESS_CODE:
                    st.session_state.password_entered = True
                    st.rerun()
                else:
                    st.error("Kode akses salah. Silakan coba lagi.")
    return False


if not check_password():
    st.stop()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1.5rem 0;">
        <p style="letter-spacing:3px; font-size:0.8rem; color:#9aa0a6; margin-bottom:0.2rem;">PULCRA · RADIX</p>
        <h1 style="margin:0;">🧵 Analisis Abrasi Benang</h1>
        <p style="color:#9aa0a6; margin-top:0.3rem;">Visualisasi data dan perhitungan titik potong pada X = {x}</p>
    </div>
    """.format(x=TARGET_X_VALUE),
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Kalkulasi (di-cache supaya tidak dihitung ulang setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Menghitung analisis data...")
def calculate_lines_and_points(x_values_series, y_values_series):
    results = {
        "y_at_x_50_original_curve": np.nan,
        "specific_x1_pt10_20": np.nan, "specific_y1_pt10_20": np.nan,
        "specific_x2_pt10_20": np.nan, "specific_y2_pt10_20": np.nan,
        "y_at_x_50_pt10_20_line": np.nan,
        "pt10_20_line_x_range": np.array([]), "pt10_20_line_y": np.array([]),
        "y_at_x_50_ransac_line": np.nan,
        "ransac_line_x": np.array([]), "ransac_line_y": np.array([]),
    }

    x_np = x_values_series.values
    y_np = y_values_series.values

    if len(x_np) < 2 or len(y_np) < 2:
        st.warning("Data tidak cukup untuk analisis. Masukkan minimal 2 pasangan X dan Y.")
        return results

    # Kurva data asli (interpolasi linear)
    try:
        if not np.all(np.diff(x_np) > 0):
            st.error("Nilai 'x_values' harus monoton meningkat untuk interpolasi kurva.")
            return results
        f = interpolate.interp1d(x_np, y_np, kind="linear", fill_value="extrapolate")
        results["y_at_x_50_original_curve"] = float(f(TARGET_X_VALUE))
    except ValueError as e:
        st.warning(f"Tidak dapat melakukan interpolasi kurva asli: {e}")

    # Garis antara titik ke-10 dan ke-20 (fallback ke titik pertama/terakhir)
    if len(x_np) >= 20:
        results["specific_x1_pt10_20"], results["specific_y1_pt10_20"] = x_np[9], y_np[9]
        results["specific_x2_pt10_20"], results["specific_y2_pt10_20"] = x_np[19], y_np[19]
    elif len(x_np) >= 2:
        results["specific_x1_pt10_20"], results["specific_y1_pt10_20"] = x_np[0], y_np[0]
        results["specific_x2_pt10_20"], results["specific_y2_pt10_20"] = x_np[-1], y_np[-1]

    x1, x2 = results["specific_x1_pt10_20"], results["specific_x2_pt10_20"]
    if not np.isnan(x1) and not np.isnan(x2) and x1 != x2:
        y1, y2 = results["specific_y1_pt10_20"], results["specific_y2_pt10_20"]
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        results["y_at_x_50_pt10_20_line"] = slope * TARGET_X_VALUE + intercept

        x_min, x_max = x_np.min(), x_np.max()
        results["pt10_20_line_x_range"] = np.linspace(min(x_min, TARGET_X_VALUE), max(x_max, TARGET_X_VALUE), 100)
        results["pt10_20_line_y"] = slope * results["pt10_20_line_x_range"] + intercept

    # Regresi robust RANSAC
    try:
        X_reshaped = x_np.reshape(-1, 1)
        residual_threshold = np.std(y_np) * 0.5 if np.std(y_np) > 0 else 1.0
        ransac = RANSACRegressor(
            LinearRegression(), min_samples=2, residual_threshold=residual_threshold,
            random_state=42, max_trials=1000,
        )
        ransac.fit(X_reshaped, y_np)
        results["y_at_x_50_ransac_line"] = ransac.predict(np.array([[TARGET_X_VALUE]]))[0]

        x_min, x_max = x_np.min(), x_np.max()
        results["ransac_line_x"] = np.linspace(min(x_min, TARGET_X_VALUE), max(x_max, TARGET_X_VALUE), 100)
        results["ransac_line_y"] = ransac.predict(results["ransac_line_x"].reshape(-1, 1))
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghitung regresi RANSAC: {e}")

    return results


# ----------------------------------------------------------------------------
# Palet warna konsisten untuk grafik & hasil
# ----------------------------------------------------------------------------
COLOR_DATA = "#5B8DEF"
COLOR_TARGET_LINE = "#FF6B6B"
COLOR_PT10_20 = "#F4B740"
COLOR_RANSAC = "#5FD068"

# Tema latar belakang grafik. "dark" dipakai untuk tampilan di layar (menyatu
# dengan tema aplikasi), sedangkan "light" dipakai sebagai opsi saat mengunduh.
CHART_THEMES = {
    "dark": dict(
        plot_bgcolor="#1A1D24",
        paper_bgcolor="#1A1D24",
        font_color="#E8E8E8",
        gridcolor="#2A2E37",
    ),
    "light": dict(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color="#1A1D24",
        gridcolor="#E2E2E2",
    ),
}


def create_abrasion_plot(x_values, y_values, results, analysis_choice, background_mode="dark"):
    fig = go.Figure()
    if x_values.empty or y_values.empty:
        return fig

    theme = CHART_THEMES.get(background_mode, CHART_THEMES["dark"])

    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode="lines+markers", name="Data Abrasi",
        line=dict(color=COLOR_DATA, width=3), marker=dict(size=7, color=COLOR_DATA),
    ))

    y_min, y_max = y_values.min(), y_values.max()
    span = y_max - y_min
    y0 = y_min - span * 0.1 if span > 0 else 0
    y1 = y_max + span * 0.1 if span > 0 else 1000
    fig.add_shape(type="line", x0=TARGET_X_VALUE, y0=y0, x1=TARGET_X_VALUE, y1=y1,
                  line=dict(color=COLOR_TARGET_LINE, width=2, dash="dash"), layer="below")
    fig.add_annotation(x=TARGET_X_VALUE, y=y1, text=f"x = {TARGET_X_VALUE}", showarrow=False,
                        yanchor="bottom", font=dict(color=COLOR_TARGET_LINE, size=13))

    show_pt10_20 = analysis_choice in ("Garis Titik 10 & 20", "Tampilkan Semua")
    show_ransac = analysis_choice in ("Garis yang melewati banyak titik", "Tampilkan Semua")
    show_original = analysis_choice in ("Kurva Data Asli", "Tampilkan Semua")

    if show_pt10_20 and results.get("pt10_20_line_x_range", np.array([])).size > 0:
        fig.add_trace(go.Scatter(x=results["pt10_20_line_x_range"], y=results["pt10_20_line_y"],
                                  mode="lines", name="Garis Titik 10 & 20",
                                  line=dict(color=COLOR_PT10_20, width=2, dash="dot")))
        if not np.isnan(results["y_at_x_50_pt10_20_line"]):
            fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results["y_at_x_50_pt10_20_line"]],
                                      mode="markers", name="Potongan Garis 10-20",
                                      marker=dict(size=12, color=COLOR_PT10_20, symbol="star")))

    if show_ransac and results.get("ransac_line_x", np.array([])).size > 0:
        fig.add_trace(go.Scatter(x=results["ransac_line_x"], y=results["ransac_line_y"],
                                  mode="lines", name="Regresi RANSAC",
                                  line=dict(color=COLOR_RANSAC, width=2, dash="dash")))
        if not np.isnan(results["y_at_x_50_ransac_line"]):
            fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results["y_at_x_50_ransac_line"]],
                                      mode="markers", name="Potongan RANSAC",
                                      marker=dict(size=12, color=COLOR_RANSAC, symbol="star")))

    if show_original and not np.isnan(results.get("y_at_x_50_original_curve", np.nan)):
        fig.add_trace(go.Scatter(x=[TARGET_X_VALUE], y=[results["y_at_x_50_original_curve"]],
                                  mode="markers", name="Potongan Kurva Asli",
                                  marker=dict(size=12, color=COLOR_DATA, symbol="star")))

    fig.update_layout(
        xaxis_title="Nilai X", yaxis_title="Nilai Benang Putus (N)",
        plot_bgcolor=theme["plot_bgcolor"], paper_bgcolor=theme["paper_bgcolor"],
        font=dict(color=theme["font_color"]),
        xaxis=dict(showgrid=True, gridcolor=theme["gridcolor"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=theme["gridcolor"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=10, r=10, b=10, t=40),
        height=480,
    )
    return fig


# ----------------------------------------------------------------------------
# Input data
# ----------------------------------------------------------------------------
st.subheader("1. Input Data")
tab_manual, tab_excel = st.tabs(["✍️ Input Manual", "📁 Impor dari Excel"])

with tab_manual:
    st.caption("Nilai X tetap dan tidak dapat diubah. Edit kolom 'Nilai Benang Putus (N)' sesuai kebutuhan.")

    edited_data = pd.DataFrame({
        "x_value": st.session_state.data["x_values"],
        "y_value": st.session_state.data["y_values"],
    })
    edited_data.index = edited_data.index + 1

    edited_df = st.data_editor(
        edited_data,
        disabled=["x_value"],
        hide_index=False,
        column_config={
            "x_value": st.column_config.NumberColumn("Nilai Tetap (X)", format="%.1f"),
            "y_value": st.column_config.NumberColumn("Nilai Benang Putus (N)", format="%.2f"),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Terapkan Perubahan", use_container_width=True):
            cleaned = edited_df.dropna(subset=["x_value", "y_value"])
            if cleaned.empty:
                st.warning("Tabel data kosong. Harap masukkan data.")
            elif not np.all(np.diff(cleaned["x_value"].values) > 0):
                st.error("Nilai 'X' harus monoton meningkat. Harap perbaiki data Anda.")
            elif len(cleaned) != len(INITIAL_DATA["x_values"]):
                st.warning("Jumlah baris berubah dari struktur standar — data dikembalikan ke nilai awal. "
                            "Gunakan tab impor Excel untuk struktur data yang berbeda.")
                st.session_state.data = pd.DataFrame(INITIAL_DATA)
            else:
                st.session_state.data = pd.DataFrame({
                    "x_values": cleaned["x_value"].values,
                    "y_values": cleaned["y_value"].values,
                })
                st.success("Data berhasil diperbarui.")
    with col2:
        if st.button("↺ Reset ke Data Awal", use_container_width=True):
            st.session_state.data = pd.DataFrame(INITIAL_DATA)
            st.success("Data telah direset.")

with tab_excel:
    st.caption("Unggah file Excel (.xlsx / .xls) dengan kolom 'x_values' dan 'y_values'.")
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_excel(uploaded_file)
            if {"x_values", "y_values"}.issubset(df_uploaded.columns):
                df_uploaded["x_values"] = pd.to_numeric(df_uploaded["x_values"], errors="coerce")
                df_uploaded["y_values"] = pd.to_numeric(df_uploaded["y_values"], errors="coerce")
                df_uploaded.dropna(subset=["x_values", "y_values"], inplace=True)

                if df_uploaded.empty:
                    st.warning("File tidak mengandung data valid setelah pembersihan.")
                elif not np.all(np.diff(df_uploaded["x_values"].values) > 0):
                    st.error("Nilai 'x_values' harus monoton meningkat.")
                else:
                    st.session_state.data = df_uploaded[["x_values", "y_values"]].reset_index(drop=True)
                    st.success("Data berhasil diimpor.")
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
            else:
                st.error("File Excel harus memiliki kolom 'x_values' dan 'y_values'.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")


# ----------------------------------------------------------------------------
# Analisis & visualisasi
# ----------------------------------------------------------------------------
st.divider()
st.subheader("2. Analisis & Visualisasi")

results = calculate_lines_and_points(st.session_state.data["x_values"], st.session_state.data["y_values"])

analysis_choice = st.radio(
    "Pilih jenis garis analisis:",
    ("Kurva Data Asli", "Garis Titik 10 & 20", "Garis yang melewati banyak titik", "Tampilkan Semua"),
    horizontal=True,
)

# Grafik yang ditampilkan di layar selalu memakai tema gelap, menyatu dengan tema aplikasi.
fig = create_abrasion_plot(
    st.session_state.data["x_values"], st.session_state.data["y_values"], results, analysis_choice,
    background_mode="dark",
)
st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# Hasil perhitungan
# ----------------------------------------------------------------------------
def fmt(value) -> str:
    return f"{value:.2f} N" if value is not None and not (isinstance(value, float) and np.isnan(value)) else "—"


st.subheader(f"3. Hasil Perhitungan Titik Potong di X = {TARGET_X_VALUE}")

METRIC_INFO = {
    "Kurva Data Asli": ("y_at_x_50_original_curve", "Kurva Data Asli", "Interpolasi linear pada X = 50"),
    "Garis Titik 10 & 20": ("y_at_x_50_pt10_20_line", "Garis Titik 10 & 20", "Garis lurus melalui titik ke-10 dan ke-20"),
    "Garis yang melewati banyak titik": ("y_at_x_50_ransac_line", "Garis RANSAC", "Regresi robust, tahan terhadap outlier"),
}

if analysis_choice == "Tampilkan Semua":
    cols = st.columns(3)
    for col, (key, label, caption) in zip(cols, METRIC_INFO.values()):
        with col, st.container(border=True):
            st.metric(label, fmt(results.get(key)))
            st.caption(caption)
else:
    key, label, caption = METRIC_INFO[analysis_choice]
    _, mid, _ = st.columns([1, 1, 1])
    with mid, st.container(border=True):
        st.metric(label, fmt(results.get(key)))
        st.caption(caption)


# ----------------------------------------------------------------------------
# Unduh grafik sebagai gambar (PNG), dengan pilihan warna latar
# ----------------------------------------------------------------------------
st.divider()
st.subheader("4. Unduh Grafik")
st.caption("Pilih warna latar belakang grafik sebelum mengunduhnya sebagai gambar PNG.")

bg_label = st.radio(
    "Warna latar grafik untuk diunduh:",
    ("Hitam", "Putih"),
    horizontal=True,
    key="chart_bg_choice",
)
background_mode = "dark" if bg_label == "Hitam" else "light"

# Grafik versi unduhan (mengikuti pilihan warna latar di atas). Figure ini juga
# dipakai untuk gambar di dalam dokumen Word pada bagian 5.
download_fig = create_abrasion_plot(
    st.session_state.data["x_values"], st.session_state.data["y_values"], results, analysis_choice,
    background_mode=background_mode,
)

if not KALEIDO_AVAILABLE:
    st.info("Fitur unduh grafik sebagai gambar membutuhkan paket `kaleido`. Tambahkan ke requirements.txt untuk mengaktifkan fitur ini.")
else:
    try:
        img_bytes = download_fig.to_image(format="png", scale=2, width=1200, height=650)
        st.download_button(
            f"⬇️ Unduh Grafik PNG (Latar {bg_label})",
            data=img_bytes,
            file_name=f"grafik_abrasi_latar_{bg_label.lower()}.png",
            mime="image/png",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Tidak dapat membuat gambar grafik: {e}")


# ----------------------------------------------------------------------------
# Unduh hasil sebagai dokumen Word
# ----------------------------------------------------------------------------
st.divider()
st.subheader("5. Unduh Hasil Analisis (Dokumen Word)")

if not DOCX_AVAILABLE:
    st.info("Fitur ekspor Word membutuhkan paket `python-docx`. Tambahkan ke requirements.txt untuk mengaktifkan fitur ini.")
else:
    filename = st.text_input("Nama file (tanpa ekstensi .docx)", value="Hasil_Analisis_Abrasi")

    if st.button("📄 Buat Dokumen Word"):
        if not filename.strip():
            st.warning("Masukkan nama file terlebih dahulu.")
        else:
            doc = Document()
            doc.add_heading("Hasil Analisis Abrasi Benang", level=1)
            doc.add_paragraph(f"Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            doc.add_heading("Data Abrasi", level=2)
            table = doc.add_table(rows=1, cols=2)
            hdr = table.rows[0].cells
            hdr[0].text, hdr[1].text = "Nilai X", "Nilai Benang Putus (N)"
            for x, y in zip(st.session_state.data["x_values"], st.session_state.data["y_values"]):
                row = table.add_row().cells
                row[0].text, row[1].text = str(x), str(y)

            if KALEIDO_AVAILABLE:
                doc.add_heading("Grafik Analisis", level=2)
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                        # Memakai grafik dengan latar yang sama seperti dipilih di bagian 4
                        # (latar putih umumnya lebih cocok untuk dicetak/dilampirkan).
                        download_fig.write_image(tmp_img.name)
                        doc.add_picture(tmp_img.name, width=Inches(6))
                except Exception as e:
                    doc.add_paragraph(f"(Grafik tidak dapat disertakan: {e})")
            else:
                doc.add_paragraph("(Paket 'kaleido' belum terinstal sehingga grafik tidak disertakan.)")

            doc.add_heading("Hasil Perhitungan", level=2)
            doc.add_paragraph(f"Nilai perpotongan pada X = {TARGET_X_VALUE}:")
            keys_to_show = (
                METRIC_INFO.values() if analysis_choice == "Tampilkan Semua" else [METRIC_INFO[analysis_choice]]
            )
            for key, label, _ in keys_to_show:
                val = results.get(key)
                if val is not None and not np.isnan(val):
                    doc.add_paragraph(f"{label}: {val:.2f} N", style="List Bullet")

            buf = io.BytesIO()
            doc.save(buf)
            buf.seek(0)

            st.success("Dokumen siap diunduh.")
            st.download_button(
                "⬇️ Unduh Dokumen Word",
                data=buf,
                file_name=f"{filename}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.caption("Aplikasi Analisis Abrasi Benang — dibuat oleh RADIX")
