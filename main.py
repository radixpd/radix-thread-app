import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import interpolate, stats
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import io
import base64
from datetime import datetime
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Thread Abrasion Analyzer by PULCRA",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Enhanced CSS with Better Responsiveness ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary-color: #8B4513;
        --secondary-color: #DAA520;
        --accent-color: #FF6B35;
        --background-dark: #0A0A0A;
        --surface-dark: #1A1A1A;
        --surface-light: #2A2A2A;
        --text-primary: #F8F8F8;
        --text-secondary: #E0E0E0;
        --text-muted: #B0B0B0;
        --border-color: #282828;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
    }
    
    /* Base Styles */
    html, body {
        background-color: var(--background-dark) !important;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: var(--background-dark) !important;
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .main {
        background-color: var(--background-dark);
        color: var(--text-secondary);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        font-family: 'Playfair Display', serif;
        font-weight: 600;
    }
    
    h1 {
        font-size: clamp(28px, 5vw, 44px);
        text-align: center;
        margin-bottom: 20px;
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: clamp(20px, 4vw, 32px);
        color: var(--secondary-color);
        border-bottom: 2px solid var(--border-color);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    h3 {
        font-size: clamp(16px, 3vw, 24px);
        margin: 20px 0 15px 0;
    }
    
    /* Enhanced Cards */
    .enhanced-card {
        background: linear-gradient(145deg, var(--surface-dark), var(--surface-light));
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .enhanced-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    
    /* Stats Cards */
    .stats-card {
        background: linear-gradient(135deg, var(--surface-dark), var(--surface-light));
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid var(--border-color);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        border-color: var(--secondary-color);
        box-shadow: 0 8px 25px rgba(218, 165, 32, 0.2);
    }
    
    .stats-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--secondary-color);
        margin-bottom: 8px;
    }
    
    .stats-label {
        font-size: 14px;
        color: var(--text-muted);
        font-weight: 500;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 69, 19, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 69, 19, 0.4);
        background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--surface-dark);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: var(--text-muted);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--secondary-color);
        color: white;
    }
    
    /* Enhanced Radio Buttons */
    .stRadio > div {
        background-color: var(--surface-dark);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid var(--border-color);
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: center;
    }
    
    .stRadio [data-baseweb="radio"] {
        background-color: var(--surface-light);
        border-radius: 8px;
        padding: 12px 20px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stRadio [data-baseweb="radio"]:hover {
        border-color: var(--secondary-color);
        background-color: rgba(218, 165, 32, 0.1);
    }
    
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background-color: var(--secondary-color);
        border-color: var(--secondary-color);
        box-shadow: 0 4px 15px rgba(218, 165, 32, 0.3);
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, var(--surface-dark), var(--background-dark));
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 40px;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    
    .pulcra-logo {
        font-family: 'Playfair Display', serif;
        font-size: clamp(32px, 8vw, 56px);
        font-weight: 700;
        background: linear-gradient(135deg, var(--secondary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
        letter-spacing: 4px;
    }
    
    /* Data Editor Enhancements */
    [data-testid="stDataEditor"] {
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* File Uploader */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--secondary-color);
        border-radius: 16px;
        background: linear-gradient(135deg, var(--surface-dark), var(--surface-light));
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-color);
        background: linear-gradient(135deg, var(--surface-light), var(--surface-dark));
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--surface-dark), var(--surface-light));
        border: 1px solid var(--border-color);
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        margin-top: 60px;
        padding: 30px;
        background: linear-gradient(135deg, var(--surface-dark), var(--background-dark));
        border-radius: 16px;
        border: 1px solid var(--border-color);
        color: var(--text-muted);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .stApp {
            padding: 10px;
        }
        
        .enhanced-card {
            padding: 16px;
            margin-bottom: 16px;
        }
        
        .app-header {
            padding: 24px;
            margin-bottom: 24px;
        }
        
        .stats-card {
            height: 120px;
            padding: 16px;
        }
        
        .stats-value {
            font-size: 24px;
        }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Constants ---
ACCESS_CODE = "RADIX2025"
TARGET_X_VALUE = 50
INITIAL_DATA = {
    'x_values': [1.7, 3.3, 5.0, 6.7, 8.4, 10.2, 12.0, 13.9, 15.8, 17.7, 19.7, 21.7, 23.8, 26.0, 28.2, 30.4, 32.8, 35.3, 37.8, 40.4, 43.3, 46.1, 49.2, 52.5, 56.0, 59.9, 64.1, 68.9, 74.66, 82.1],
    'y_values': [105, 143, 157, 185, 191, 191, 200, 250, 266, 292, 337, 343, 345, 397, 397, 404, 425, 457, 476, 476, 501, 535, 555, 623, 623, 635, 667, 770, 805, 974]
}

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(INITIAL_DATA)
    if 'calculated_results' not in st.session_state:
        st.session_state.calculated_results = {}
    if 'password_entered' not in st.session_state:
        st.session_state.password_entered = False
    if 'data_needs_recalc' not in st.session_state:
        st.session_state.data_needs_recalc = True
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

# --- Authentication ---
def check_password():
    """Enhanced password check with better UI"""
    if not st.session_state.get('password_entered', False):
        st.markdown("""
        <div class="enhanced-card" style="max-width: 400px; margin: 100px auto; text-align: center;">
            <h2>🔐 Akses Aplikasi</h2>
            <p style="color: var(--text-muted); margin-bottom: 30px;">
                Masukkan kode akses untuk melanjutkan
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password_input = st.text_input(
                "Kode Akses", 
                type="password", 
                key="password_input",
                placeholder="Masukkan kode akses...",
                help="Hubungi administrator untuk mendapatkan kode akses"
            )
            
            if st.button("🚀 Masuk", key="login_button", use_container_width=True):
                if password_input == ACCESS_CODE:
                    st.session_state.password_entered = True
                    st.success("✅ Akses berhasil! Memuat aplikasi...")
                    st.rerun()
                else:
                    st.error("❌ Kode akses salah. Silakan coba lagi.")
        
        return False
    return True

# --- Enhanced Calculation Functions ---
@st.cache_data(show_spinner="🔄 Menghitung analisis data...")
def calculate_enhanced_analysis(x_values_series, y_values_series):
    """Enhanced calculation with additional statistical measures"""
    results = {
        'y_at_x_50_original_curve': np.nan,
        'specific_x1_pt10_20': np.nan, 'specific_y1_pt10_20': np.nan,
        'specific_x2_pt10_20': np.nan, 'specific_y2_pt10_20': np.nan,
        'y_at_x_50_pt10_20_line': np.nan,
        'pt10_20_line_x_range': np.array([]), 'pt10_20_line_y': np.array([]),
        'y_at_x_50_ransac_line': np.nan,
        'ransac_line_x': np.array([]), 'ransac_line_y': np.array([]),
        'r2_score_pt10_20': np.nan,
        'r2_score_ransac': np.nan,
        'rmse_pt10_20': np.nan,
        'rmse_ransac': np.nan,
        'confidence_interval_original': (np.nan, np.nan),
        'data_stats': {}
    }
    
    x_np = x_values_series.values
    y_np = y_values_series.values
    
    if len(x_np) < 2 or len(y_np) < 2:
        st.warning("⚠️ Data tidak cukup untuk analisis. Masukkan minimal 2 pasangan X dan Y.")
        return results
    
    # Calculate basic statistics
    results['data_stats'] = {
        'count': len(y_np),
        'mean_y': np.mean(y_np),
        'std_y': np.std(y_np),
        'min_y': np.min(y_np),
        'max_y': np.max(y_np),
        'range_x': (np.min(x_np), np.max(x_np)),
        'range_y': (np.min(y_np), np.max(y_np))
    }
    
    # Original curve interpolation with confidence interval
    try:
        if not np.all(np.diff(x_np) > 0):
            st.error("❌ Nilai 'x_values' harus monoton meningkat untuk interpolasi kurva.")
            return results
        
        f = interpolate.interp1d(x_np, y_np, kind='linear', fill_value='extrapolate')
        results['y_at_x_50_original_curve'] = float(f(TARGET_X_VALUE))
        
        # Simple confidence interval estimation
        nearby_indices = np.where((x_np >= TARGET_X_VALUE - 5) & (x_np <= TARGET_X_VALUE + 5))[0]
        if len(nearby_indices) > 1:
            nearby_y = y_np[nearby_indices]
            std_nearby = np.std(nearby_y)
            results['confidence_interval_original'] = (
                results['y_at_x_50_original_curve'] - 1.96 * std_nearby,
                results['y_at_x_50_original_curve'] + 1.96 * std_nearby
            )
    except Exception as e:
        st.warning(f"⚠️ Tidak dapat melakukan interpolasi kurva asli: {e}")
    
    # Line between points 10 & 20 with enhanced metrics
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
            
            # Calculate R² and RMSE for pt10_20 line
            y_pred_pt10_20 = slope_pt10_20 * x_np + intercept_pt10_20
            results['r2_score_pt10_20'] = r2_score(y_np, y_pred_pt10_20)
            results['rmse_pt10_20'] = np.sqrt(mean_squared_error(y_np, y_pred_pt10_20))
    
    # Enhanced RANSAC regression
    if len(x_np) >= 2:
        try:
            X_reshaped = x_np.reshape(-1, 1)
            residual_threshold_val = np.std(y_np) * 0.5 if len(y_np) > 1 and np.std(y_np) > 0 else 1.0
            
            ransac = RANSACRegressor(
                LinearRegression(),
                min_samples=max(2, int(len(x_np) * 0.1)),
                residual_threshold=residual_threshold_val,
                random_state=42,
                max_trials=1000
            )
            ransac.fit(X_reshaped, y_np)
            results['y_at_x_50_ransac_line'] = ransac.predict(np.array([[TARGET_X_VALUE]]))[0]
            
            x_min_plot = x_np.min() if x_np.size > 0 else 0
            x_max_plot = x_np.max() if x_np.size > 0 else 100
            results['ransac_line_x'] = np.linspace(min(x_min_plot, TARGET_X_VALUE), max(x_max_plot, TARGET_X_VALUE), 100)
            results['ransac_line_y'] = ransac.predict(results['ransac_line_x'].reshape(-1, 1))
            
            # Calculate R² and RMSE for RANSAC
            y_pred_ransac = ransac.predict(X_reshaped)
            results['r2_score_ransac'] = r2_score(y_np, y_pred_ransac)
            results['rmse_ransac'] = np.sqrt(mean_squared_error(y_np, y_pred_ransac))
            
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat menghitung regresi RANSAC: {e}")
    
    return results

# --- Enhanced Plotting Function ---
def create_enhanced_plot(x_values, y_values, results, analysis_choice):
    """Create enhanced plot with better styling and interactivity"""
    fig = go.Figure()
    
    # Add main data curve
    if not x_values.empty and not y_values.empty:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name='📊 Data Abrasi',
            line=dict(color='#8B4513', width=3, shape='spline'),
            marker=dict(
                size=8, 
                color='#DAA520',
                line=dict(width=2, color='#8B4513')
            ),
            hovertemplate="<b>Data Point</b><br>X: %{x}<br>Y: %{y:.2f} N<extra></extra>"
        ))
        
        # Add vertical line at x=50
        plot_y_min = y_values.min() if not y_values.empty else 0
        plot_y_max = y_values.max() if not y_values.empty else 1000
        y_range_span = plot_y_max - plot_y_min
        y0_line = plot_y_min - y_range_span * 0.1 if y_range_span > 0 else 0
        y1_line = plot_y_max + y_range_span * 0.1 if y_range_span > 0 else 1000
        
        fig.add_shape(
            type="line",
            x0=TARGET_X_VALUE, y0=y0_line,
            x1=TARGET_X_VALUE, y1=y1_line,
            line=dict(color="#FF6B35", width=3, dash="dash"),
            layer="below"
        )
        
        fig.add_annotation(
            x=TARGET_X_VALUE, y=y1_line * 0.95,
            text=f"🎯 X = {TARGET_X_VALUE}",
            showarrow=False,
            font=dict(color="#FF6B35", size=14, family="Inter, sans-serif"),
            bgcolor="rgba(26,26,26,0.8)",
            bordercolor="#FF6B35",
            borderwidth=2,
            borderpad=6
        )
        
        # Add analysis-specific traces
        if analysis_choice in ["Garis Titik 10 & 20", "Tampilkan Semua"]:
            if results.get('pt10_20_line_x_range', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['pt10_20_line_x_range'],
                    y=results['pt10_20_line_y'],
                    mode='lines',
                    name='📈 Garis Titik 10 & 20',
                    line=dict(color='#ADD8E6', width=3, dash='dot')
                ))
                
                # Add specific points
                if not np.isnan(results['specific_x1_pt10_20']):
                    fig.add_trace(go.Scatter(
                        x=[results['specific_x1_pt10_20']],
                        y=[results['specific_y1_pt10_20']],
                        mode='markers',
                        name='🔟 Titik ke-10',
                        marker=dict(size=12, color='#ADD8E6', symbol='circle', line=dict(width=2, color='white'))
                    ))
                
                if not np.isnan(results['specific_x2_pt10_20']):
                    fig.add_trace(go.Scatter(
                        x=[results['specific_x2_pt10_20']],
                        y=[results['specific_y2_pt10_20']],
                        mode='markers',
                        name='2️⃣0️⃣ Titik ke-20',
                        marker=dict(size=12, color='#ADD8E6', symbol='circle', line=dict(width=2, color='white'))
                    ))
            
            if not np.isnan(results.get('y_at_x_50_pt10_20_line')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE],
                    y=[results['y_at_x_50_pt10_20_line']],
                    mode='markers',
                    name='⭐ Potongan Garis 10-20',
                    marker=dict(size=15, color='#ADD8E6', symbol='star', line=dict(width=2, color='white')),
                    hovertemplate=f"<b>Potongan (Garis 10-20)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}} N<extra></extra>"
                ))
        
        if analysis_choice in ["Garis yang melewati banyak titik", "Tampilkan Semua"]:
            if results.get('ransac_line_x', []).size > 0:
                fig.add_trace(go.Scatter(
                    x=results['ransac_line_x'],
                    y=results['ransac_line_y'],
                    mode='lines',
                    name='📉 Regresi RANSAC',
                    line=dict(color='#90EE90', width=3, dash='dash')
                ))
            
            if not np.isnan(results.get('y_at_x_50_ransac_line')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE],
                    y=[results['y_at_x_50_ransac_line']],
                    mode='markers',
                    name='⭐ Potongan RANSAC',
                    marker=dict(size=15, color='#90EE90', symbol='star', line=dict(width=2, color='white')),
                    hovertemplate=f"<b>Potongan (RANSAC)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}} N<extra></extra>"
                ))
        
        if analysis_choice in ["Kurva Data Asli", "Tampilkan Semua"]:
            if not np.isnan(results.get('y_at_x_50_original_curve')):
                fig.add_trace(go.Scatter(
                    x=[TARGET_X_VALUE],
                    y=[results['y_at_x_50_original_curve']],
                    mode='markers',
                    name='⭐ Potongan Kurva Asli',
                    marker=dict(size=15, color='#DAA520', symbol='star', line=dict(width=2, color='white')),
                    hovertemplate=f"<b>Potongan (Kurva Asli)</b><br>X: {TARGET_X_VALUE}<br>Y: %{{y:.2f}} N<extra></extra>"
                ))
    
    # Enhanced layout
    fig.update_layout(
        title={
            'text': '🧵 Grafik Analisis Abrasi Benang',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Playfair Display, serif', 'color': '#F8F8F8'}
        },
        xaxis_title='Nilai X',
        yaxis_title='Nilai Benang Putus (N)',
        plot_bgcolor='#1A1A1A',
        paper_bgcolor='#1A1A1A',
        font=dict(color='#E0E0E0', family='Inter, sans-serif'),
        xaxis=dict(
            showgrid=True,
            gridcolor='#282828',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=16),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#282828',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=16),
            tickfont=dict(size=12)
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26,26,26,0.8)",
            bordercolor="#282828",
            borderwidth=1,
            font=dict(size=12)
        ),
        hovermode="x unified",
        margin=dict(l=60, r=150, b=60, t=80)
    )
    
    return fig

# --- Data Export Functions ---
def create_download_link(df, filename, file_format="csv"):
    """Create download link for data"""
    if file_format == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Download CSV</a>'
    elif file_format == "excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Download Excel</a>'
    
    return href

def export_results_to_json(results, data_stats):
    """Export analysis results to JSON"""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis_results": {
            "original_curve_y_at_50": float(results.get('y_at_x_50_original_curve', 0)) if not np.isnan(results.get('y_at_x_50_original_curve', np.nan)) else None,
            "pt10_20_line_y_at_50": float(results.get('y_at_x_50_pt10_20_line', 0)) if not np.isnan(results.get('y_at_x_50_pt10_20_line', np.nan)) else None,
            "ransac_line_y_at_50": float(results.get('y_at_x_50_ransac_line', 0)) if not np.isnan(results.get('y_at_x_50_ransac_line', np.nan)) else None,
            "r2_scores": {
                "pt10_20": float(results.get('r2_score_pt10_20', 0)) if not np.isnan(results.get('r2_score_pt10_20', np.nan)) else None,
                "ransac": float(results.get('r2_score_ransac', 0)) if not np.isnan(results.get('r2_score_ransac', np.nan)) else None
            },
            "rmse_values": {
                "pt10_20": float(results.get('rmse_pt10_20', 0)) if not np.isnan(results.get('rmse_pt10_20', np.nan)) else None,
                "ransac": float(results.get('rmse_ransac', 0)) if not np.isnan(results.get('rmse_ransac', np.nan)) else None
            }
        },
        "data_statistics": data_stats
    }
    
    json_str = json.dumps(export_data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">📥 Download Analysis Results (JSON)</a>'
    return href

# --- Main Application ---
def main():
    initialize_session_state()
    
    if not check_password():
        st.stop()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="pulcra-logo">PULCRA</div>
        <h1>🧵 Analisis Abrasi Benang</h1>
        <p style="font-size: 18px; color: var(--text-muted); margin-top: 16px;">
            Platform profesional untuk analisis data abrasi dengan visualisasi interaktif dan perhitungan statistik lanjutan
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Statistics Overview
    if not st.session_state.data.empty:
        st.markdown("### 📊 Ringkasan Data")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{len(st.session_state.data)}</div>
                <div class="stats-label">Total Data Points</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mean_y = st.session_state.data['y_values'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{mean_y:.1f}</div>
                <div class="stats-label">Rata-rata Y</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            std_y = st.session_state.data['y_values'].std()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{std_y:.1f}</div>
                <div class="stats-label">Std Deviasi Y</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            min_y = st.session_state.data['y_values'].min()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{min_y:.1f}</div>
                <div class="stats-label">Min Y</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            max_y = st.session_state.data['y_values'].max()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{max_y:.1f}</div>
                <div class="stats-label">Max Y</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Data Input Section
    st.markdown("### 📝 Input Data")
    st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
    
    tabs = st.tabs(["✏️ Input Manual", "📁 Impor Excel", "📤 Export Data"])
    
    with tabs[0]:
        st.markdown("**Masukkan data abrasi ke tabel. Nilai X adalah tetap dan tidak dapat diubah.**")
        
        edited_data = pd.DataFrame({
            'x_value': st.session_state.data['x_values'],
            'y_value': st.session_state.data['y_values']
        })
        edited_data.index = edited_data.index + 1
        
        edited_df = st.data_editor(
            edited_data,
            disabled=["x_value"],
            hide_index=False,
            column_config={
                "x_value": st.column_config.NumberColumn(
                    "Nilai X (Tetap)", 
                    format="%.1f", 
                    help="Nilai X standar yang tidak dapat diubah"
                ),
                "y_value": st.column_config.NumberColumn(
                    "Nilai Benang Putus (N)", 
                    format="%.2f", 
                    help="Nilai gaya putus benang dalam Newton"
                ),
            },
            num_rows="dynamic",
            use_container_width=True,
            key="data_editor",
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Terapkan Perubahan", key="apply_changes", use_container_width=True):
                try:
                    cleaned_edited_df = edited_df.dropna(subset=['x_value', 'y_value'])
                    if cleaned_edited_df.empty:
                        st.warning("⚠️ Tabel data kosong. Harap masukkan data.")
                    elif not np.all(np.diff(cleaned_edited_df['x_value'].values) > 0):
                        st.error("❌ Nilai 'x_value' harus monoton meningkat.")
                    else:
                        st.session_state.data = pd.DataFrame({
                            'x_values': cleaned_edited_df['x_value'].values,
                            'y_values': cleaned_edited_df['y_value'].values
                        })
                        st.session_state.data_needs_recalc = True
                        st.success("✅ Data berhasil diperbarui!")
                except Exception as e:
                    st.error(f"❌ Kesalahan: {e}")
        
        with col2:
            if st.button("🔄 Reset ke Data Awal", key="reset_data", use_container_width=True):
                st.session_state.data = pd.DataFrame(INITIAL_DATA)
                st.session_state.data_needs_recalc = True
                st.success("✅ Data telah direset!")
        
        with col3:
            if st.button("📊 Tampilkan Statistik", key="show_stats", use_container_width=True):
                if not st.session_state.data.empty:
                    st.dataframe(st.session_state.data.describe(), use_container_width=True)
    
    with tabs[1]:
        st.markdown("**Unggah file Excel dengan kolom 'x_values' dan 'y_values'**")
        
        uploaded_file = st.file_uploader(
            "Pilih file Excel", 
            type=["xlsx", "xls"], 
            key="file_uploader",
            help="File harus berisi kolom 'x_values' dan 'y_values'"
        )
        
        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_excel(uploaded_file)
                
                if 'x_values' in df_uploaded.columns and 'y_values' in df_uploaded.columns:
                    df_uploaded['x_values'] = pd.to_numeric(df_uploaded['x_values'], errors='coerce')
                    df_uploaded['y_values'] = pd.to_numeric(df_uploaded['y_values'], errors='coerce')
                    df_uploaded.dropna(subset=['x_values', 'y_values'], inplace=True)
                    
                    if df_uploaded.empty:
                        st.warning("⚠️ File kosong atau tidak valid.")
                    elif not np.all(np.diff(df_uploaded['x_values'].values) > 0):
                        st.error("❌ Nilai 'x_values' harus monoton meningkat.")
                    else:
                        st.session_state.data = df_uploaded[['x_values', 'y_values']]
                        st.session_state.data_needs_recalc = True
                        st.success("✅ Data Excel berhasil diimpor!")
                        
                        # Show preview
                        st.markdown("**Preview Data:**")
                        st.dataframe(st.session_state.data.head(10), use_container_width=True)
                else:
                    st.error("❌ File harus mengandung kolom 'x_values' dan 'y_values'.")
            except Exception as e:
                st.error(f"❌ Kesalahan membaca file: {e}")
    
    with tabs[2]:
        st.markdown("**Export data dan hasil analisis**")
        
        if not st.session_state.data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Export:**")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # CSV Download
                csv_link = create_download_link(st.session_state.data, f"abrasion_data_{timestamp}", "csv")
                st.markdown(csv_link, unsafe_allow_html=True)
                
                # Excel Download
                excel_link = create_download_link(st.session_state.data, f"abrasion_data_{timestamp}", "excel")
                st.markdown(excel_link, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Analysis Results Export:**")
                if st.session_state.calculated_results:
                    json_link = export_results_to_json(
                        st.session_state.calculated_results,
                        st.session_state.calculated_results.get('data_stats', {})
                    )
                    st.markdown(json_link, unsafe_allow_html=True)
                else:
                    st.info("ℹ️ Lakukan analisis terlebih dahulu untuk export hasil.")
        else:
            st.info("ℹ️ Tidak ada data untuk di-export.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis Section
    st.markdown("### 🔬 Analisis & Visualisasi")
    
    # Calculate if needed
    if st.session_state.data_needs_recalc and not st.session_state.data.empty:
        with st.spinner("🔄 Menghitung analisis..."):
            st.session_state.calculated_results = calculate_enhanced_analysis(
                st.session_state.data['x_values'],
                st.session_state.data['y_values']
            )
            st.session_state.data_needs_recalc = False
    
    # Analysis Choice
    st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
    st.markdown("**Pilih jenis analisis yang ingin ditampilkan:**")
    
    analysis_choice = st.radio(
        "",
        ("🔵 Kurva Data Asli", "📈 Garis Titik 10 & 20", "📉 Garis RANSAC", "🌟 Tampilkan Semua"),
        key="analysis_choice_radio",
        horizontal=True
    )
    
    # Clean up choice for processing
    analysis_choice_clean = analysis_choice.split(" ", 1)[1] if " " in analysis_choice else analysis_choice
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Plot
    if not st.session_state.data.empty and st.session_state.calculated_results:
        fig = create_enhanced_plot(
            st.session_state.data['x_values'],
            st.session_state.data['y_values'],
            st.session_state.calculated_results,
            analysis_choice_clean
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Results Section
        st.markdown("### 🎯 Hasil Perhitungan di X = 50")
        
        if analysis_choice_clean == "Kurva Data Asli":
            st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 🔵 Kurva Data Asli")
                if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
                    value = st.session_state.calculated_results['y_at_x_50_original_curve']
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <div style="font-size: 48px; font-weight: bold; color: #DAA520; margin-bottom: 10px;">
                            {value:.2f} N
                        </div>
                        <div style="color: var(--text-muted); font-style: italic;">
                            Interpolasi linear dari kurva data asli pada X=50
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence interval if available
                    ci = st.session_state.calculated_results.get('confidence_interval_original')
                    if not (np.isnan(ci[0]) or np.isnan(ci[1])):
                        st.info(f"📊 Interval Kepercayaan 95%: {ci[0]:.2f} - {ci[1]:.2f} N")
                else:
                    st.warning("⚠️ Tidak dapat dihitung")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_choice_clean == "Garis Titik 10 & 20":
            st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📈 Garis Titik 10 & 20")
                if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
                    value = st.session_state.calculated_results['y_at_x_50_pt10_20_line']
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <div style="font-size: 48px; font-weight: bold; color: #ADD8E6; margin-bottom: 10px;">
                            {value:.2f} N
                        </div>
                        <div style="color: var(--text-muted); font-style: italic;">
                            Regresi linear melalui titik ke-10 dan ke-20
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak dapat dihitung")
            
            with col2:
                if not np.isnan(st.session_state.calculated_results.get('r2_score_pt10_20')):
                    r2 = st.session_state.calculated_results['r2_score_pt10_20']
                    rmse = st.session_state.calculated_results.get('rmse_pt10_20', 0)
                    st.metric("R² Score", f"{r2:.3f}")
                    st.metric("RMSE", f"{rmse:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_choice_clean == "Garis RANSAC":
            st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("#### 📉 Garis RANSAC")
                if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
                    value = st.session_state.calculated_results['y_at_x_50_ransac_line']
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px;">
                        <div style="font-size: 48px; font-weight: bold; color: #90EE90; margin-bottom: 10px;">
                            {value:.2f} N
                        </div>
                        <div style="color: var(--text-muted); font-style: italic;">
                            Regresi robust RANSAC, tahan terhadap outlier
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak dapat dihitung")
            
            with col2:
                if not np.isnan(st.session_state.calculated_results.get('r2_score_ransac')):
                    r2 = st.session_state.calculated_results['r2_score_ransac']
                    rmse = st.session_state.calculated_results.get('rmse_ransac', 0)
                    st.metric("R² Score", f"{r2:.3f}")
                    st.metric("RMSE", f"{rmse:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif analysis_choice_clean == "Tampilkan Semua":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="enhanced-card" style="height: 280px;">', unsafe_allow_html=True)
                st.markdown("#### 🔵 Kurva Asli")
                if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_original_curve')):
                    value = st.session_state.calculated_results['y_at_x_50_original_curve']
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px;">
                        <div style="font-size: 36px; font-weight: bold; color: #DAA520; margin-bottom: 8px;">
                            {value:.2f} N
                        </div>
                        <div style="color: var(--text-muted); font-size: 14px; font-style: italic;">
                            Interpolasi kurva asli
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak dapat dihitung")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="enhanced-card" style="height: 280px;">', unsafe_allow_html=True)
                st.markdown("#### 📈 Garis 10-20")
                if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_pt10_20_line')):
                    value = st.session_state.calculated_results['y_at_x_50_pt10_20_line']
                    r2 = st.session_state.calculated_results.get('r2_score_pt10_20', 0)
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px;">
                        <div style="font-size: 36px; font-weight: bold; color: #ADD8E6; margin-bottom: 8px;">
                            {value:.2f} N
                        </div>
                        <div style="color: var(--text-muted); font-size: 14px; font-style: italic;">
                            Regresi titik 10 & 20
                        </div>
                        <div style="color: var(--text-muted); font-size: 12px; margin-top: 10px;">
                            R² = {r2:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak dapat dihitung")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="enhanced-card" style="height: 280px;">', unsafe_allow_html=True)
                st.markdown("#### 📉 RANSAC")
                if not np.isnan(st.session_state.calculated_results.get('y_at_x_50_ransac_line')):
                    value = st.session_state.calculated_results['y_at_x_50_ransac_line']
                    r2 = st.session_state.calculated_results.get('r2_score_ransac', 0)
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px;">
                        <div style="font-size: 36px; font-weight: bold; color: #90EE90; margin-bottom: 8px;">
                            {value:.2f} N
                        </div>
                        <div style="color: var(--text-muted); font-size: 14px; font-style: italic;">
                            Regresi robust
                        </div>
                        <div style="color: var(--text-muted); font-size: 12px; margin-top: 10px;">
                            R² = {r2:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Tidak dapat dihitung")
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("ℹ️ Masukkan data untuk melihat analisis dan visualisasi.")
    
    # Footer
    st.markdown("""
    <div class="app-footer">
        <h3 style="margin-bottom: 20px; color: var(--secondary-color);">🧵 Thread Abrasion Analyzer</h3>
        <p>Dikembangkan dengan ❤️ oleh <strong>PULCRA</strong></p>
        <p style="font-size: 14px; margin-top: 15px; color: var(--text-muted);">
            Platform analisis profesional untuk pengujian abrasi benang dengan teknologi machine learning
        </p>
        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--border-color);">
            <small>© 2025 PULCRA. All rights reserved.</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
