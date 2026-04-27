import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decline Curve Analysis | Portfolio",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    
    .main { background-color: #0a0e17; }
    
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        color: #f9fafb;
    }
    .metric-unit {
        font-size: 12px;
        color: #9ca3af;
        margin-left: 4px;
    }
    .metric-delta {
        font-size: 12px;
        color: #10b981;
        margin-top: 4px;
    }
    
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #f59e0b;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
    .badge-exp { background: #1e3a5f; color: #60a5fa; }
    .badge-hyp { background: #1a3a2a; color: #34d399; }
    .badge-har { background: #3b2a1a; color: #fb923c; }
    
    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1f2937;
    }
    
    .stSlider label { 
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        color: #9ca3af !important;
    }
    
    h1, h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton > button {
        background: #f59e0b !important;
        color: #000 !important;
        border: none !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
    }
    
    .info-box {
        background: #0f1d35;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #93c5fd;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Arps Decline Functions ─────────────────────────────────────────────────────
def arps_exponential(t, qi, di):
    """Arps exponential decline: q(t) = qi * exp(-di * t)"""
    return qi * np.exp(-di * t)

def arps_hyperbolic(t, qi, di, b):
    """Arps hyperbolic decline: q(t) = qi / (1 + b*di*t)^(1/b)"""
    return qi / (1 + b * di * t) ** (1 / b)

def arps_harmonic(t, qi, di):
    """Arps harmonic decline (b=1): q(t) = qi / (1 + di * t)"""
    return qi / (1 + di * t)

def cumulative_exp(t, qi, di):
    return (qi / di) * (1 - np.exp(-di * t))

def cumulative_hyp(t, qi, di, b):
    return (qi**b / ((1 - b) * di)) * (qi**(1-b) - arps_hyperbolic(t, qi, di, b)**(1-b))

def cumulative_har(t, qi, di):
    return (qi / di) * np.log(1 + di * t)

def fit_decline(t_hist, q_hist, model='hyperbolic'):
    """Fit Arps decline parameters using least squares."""
    qi_init = q_hist[0]
    try:
        if model == 'exponential':
            popt, pcov = curve_fit(arps_exponential, t_hist, q_hist,
                                   p0=[qi_init, 0.05], bounds=([0, 1e-6], [np.inf, 1.0]),
                                   maxfev=5000)
            return {'qi': popt[0], 'di': popt[1], 'b': 0.0, 'pcov': pcov}
        elif model == 'hyperbolic':
            popt, pcov = curve_fit(arps_hyperbolic, t_hist, q_hist,
                                   p0=[qi_init, 0.05, 1.0], bounds=([0, 1e-6, 0.01], [np.inf, 1.0, 2.0]),
                                   maxfev=10000)
            return {'qi': popt[0], 'di': popt[1], 'b': popt[2], 'pcov': pcov}
        elif model == 'harmonic':
            popt, pcov = curve_fit(arps_harmonic, t_hist, q_hist,
                                   p0=[qi_init, 0.05], bounds=([0, 1e-6], [np.inf, 1.0]),
                                   maxfev=5000)
            return {'qi': popt[0], 'di': popt[1], 'b': 1.0, 'pcov': pcov}
    except Exception as e:
        return None

def r_squared(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1 - ss_res / ss_tot

# ─── Sample Data Generator ──────────────────────────────────────────────────────
def generate_sample_data(qi=1200, di=0.08, b=0.8, months=36, noise_pct=0.05, seed=42):
    np.random.seed(seed)
    t = np.arange(months)
    q_true = arps_hyperbolic(t, qi, di, b)
    noise = np.random.normal(0, noise_pct * q_true, len(t))
    q_noisy = np.maximum(q_true + noise, 10)
    df = pd.DataFrame({'Month': t + 1, 'Time_months': t, 'Rate_bopd': q_noisy.round(1)})
    return df

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛢️ DCA Dashboard")
    st.markdown("*Decline Curve Analysis*")
    st.markdown("---")
    
    st.markdown("### Data Input")
    data_source = st.radio("", ["Use sample data", "Upload CSV"], label_visibility="collapsed")
    
    df_input = None
    if data_source == "Use sample data":
        st.markdown('<div class="info-box">Sample: Hyperbolic decline well with noise added</div>', 
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            s_qi = st.number_input("qi (bopd)", value=1200, min_value=10)
            s_di = st.number_input("di (month⁻¹)", value=0.08, min_value=0.001, format="%.3f")
        with col2:
            s_b = st.number_input("b factor", value=0.8, min_value=0.01, max_value=2.0, step=0.1)
            s_months = st.number_input("History (months)", value=36, min_value=6)
        df_input = generate_sample_data(s_qi, s_di, s_b, s_months)
    else:
        uploaded = st.file_uploader("Upload CSV (columns: Month, Rate_bopd)", type=['csv'])
        if uploaded:
            df_input = pd.read_csv(uploaded)
            # Try to auto-detect columns
            rate_col = [c for c in df_input.columns if any(k in c.lower() for k in ['rate','oil','q','bopd'])]
            if rate_col:
                df_input = df_input.rename(columns={rate_col[0]: 'Rate_bopd'})
            df_input['Time_months'] = np.arange(len(df_input))
    
    st.markdown("---")
    st.markdown("### Forecast Settings")
    forecast_months = st.slider("Forecast horizon (months)", 12, 120, 60, 6)
    q_aban = st.number_input("Abandonment rate (bopd)", value=50, min_value=1)
    oil_price = st.number_input("Oil price (USD/bbl)", value=75.0, step=1.0)
    
    st.markdown("---")
    st.markdown("### Model Selection")
    show_exp = st.checkbox("Exponential", value=True)
    show_hyp = st.checkbox("Hyperbolic", value=True)
    show_har = st.checkbox("Harmonic", value=True)

# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown("# Decline Curve Analysis")
st.markdown("*Arps (1945) — Exponential · Hyperbolic · Harmonic*")

if df_input is None:
    st.info("Upload a CSV or use sample data from the sidebar.")
    st.stop()

t_hist = df_input['Time_months'].values.astype(float)
q_hist = df_input['Rate_bopd'].values.astype(float)
t_forecast = np.arange(0, len(t_hist) + forecast_months)

# ─── Fit All Models ────────────────────────────────────────────────────────────
models = {}
if show_exp:
    params = fit_decline(t_hist, q_hist, 'exponential')
    if params:
        q_fit = arps_exponential(t_hist, params['qi'], params['di'])
        q_fore = arps_exponential(t_forecast, params['qi'], params['di'])
        r2 = r_squared(q_hist, q_fit)
        t_aban = np.where(q_fore < q_aban)[0]
        t_aban = t_aban[0] if len(t_aban) > 0 else len(t_forecast)
        eur = cumulative_exp(min(t_aban, len(t_forecast)-1), params['qi'], params['di']) / 1000
        models['Exponential'] = {**params, 'q_fore': q_fore, 'r2': r2, 'eur_mbbl': eur,
                                   't_aban': t_aban, 'color': '#60a5fa', 'dash': 'dash'}

if show_hyp:
    params = fit_decline(t_hist, q_hist, 'hyperbolic')
    if params:
        q_fit = arps_hyperbolic(t_hist, params['qi'], params['di'], params['b'])
        q_fore = arps_hyperbolic(t_forecast, params['qi'], params['di'], params['b'])
        r2 = r_squared(q_hist, q_fit)
        t_aban = np.where(q_fore < q_aban)[0]
        t_aban = t_aban[0] if len(t_aban) > 0 else len(t_forecast)
        eur = cumulative_hyp(min(t_aban, len(t_forecast)-1), params['qi'], params['di'], params['b']) / 1000
        models['Hyperbolic'] = {**params, 'q_fore': q_fore, 'r2': r2, 'eur_mbbl': eur,
                                 't_aban': t_aban, 'color': '#34d399', 'dash': 'solid'}

if show_har:
    params = fit_decline(t_hist, q_hist, 'harmonic')
    if params:
        q_fit = arps_harmonic(t_hist, params['qi'], params['di'])
        q_fore = arps_harmonic(t_forecast, params['qi'], params['di'])
        r2 = r_squared(q_hist, q_fit)
        t_aban = np.where(q_fore < q_aban)[0]
        t_aban = t_aban[0] if len(t_aban) > 0 else len(t_forecast)
        eur = cumulative_har(min(t_aban, len(t_forecast)-1), params['qi'], params['di']) / 1000
        models['Harmonic'] = {**params, 'q_fore': q_fore, 'r2': r2, 'eur_mbbl': eur,
                               't_aban': t_aban, 'color': '#fb923c', 'dash': 'dot'}

# ─── KPI Cards ────────────────────────────────────────────────────────────────
st.markdown("---")
if models:
    best_model_name = max(models.keys(), key=lambda k: models[k]['r2'])
    best = models[best_model_name]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Fit Model", best_model_name, f"R² = {best['r2']:.4f}")
    with col2:
        st.metric("Initial Rate (qi)", f"{best['qi']:,.0f} bopd")
    with col3:
        st.metric("EUR (Best Fit)", f"{best['eur_mbbl']:,.1f} Mbbl",
                  f"≈ USD {best['eur_mbbl']*oil_price/1000:.1f}MM")
    with col4:
        st.metric("Decline Rate (di)", f"{best['di']*100:.2f}% /month",
                  f"{best['di']*1200:.1f}% /year nominal")

# ─── Main Chart ────────────────────────────────────────────────────────────────
st.markdown("---")
col_chart, col_log = st.columns([1, 1])

def make_chart(log_scale=False):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=t_hist, y=q_hist,
        mode='markers',
        name='Historical data',
        marker=dict(color='#f9fafb', size=6, symbol='circle-open', line=dict(width=1.5)),
    ))
    
    # Forecast traces
    for name, m in models.items():
        q_fore_clipped = np.where(m['q_fore'] < 1, np.nan, m['q_fore'])
        fig.add_trace(go.Scatter(
            x=t_forecast, y=q_fore_clipped,
            mode='lines',
            name=name,
            line=dict(color=m['color'], width=2, dash=m['dash']),
        ))
    
    # Abandonment line
    fig.add_hline(y=q_aban, line_dash="longdash", line_color="#6b7280", line_width=1,
                  annotation_text=f"q_aban = {q_aban} bopd", annotation_position="bottom right",
                  annotation_font=dict(color="#6b7280", size=11))
    
    # History / forecast divider
    fig.add_vline(x=t_hist[-1]+0.5, line_dash="dot", line_color="#374151", line_width=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#111827',
        plot_bgcolor='#0d1117',
        font=dict(family='IBM Plex Mono', size=11, color='#9ca3af'),
        legend=dict(
            bgcolor='#1f2937', bordercolor='#374151', borderwidth=1,
            font=dict(size=11), orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0
        ),
        xaxis=dict(title='Time (months)', gridcolor='#1f2937', showgrid=True, zeroline=False),
        yaxis=dict(title='Rate (bopd)', gridcolor='#1f2937', showgrid=True, zeroline=False,
                   type='log' if log_scale else 'linear'),
        margin=dict(t=60, b=60, l=70, r=20),
        height=380,
        hovermode='x unified',
    )
    return fig

with col_chart:
    st.markdown("**Rate vs Time — Linear Scale**")
    st.plotly_chart(make_chart(log_scale=False), use_container_width=True)

with col_log:
    st.markdown("**Rate vs Time — Log Scale**")
    st.plotly_chart(make_chart(log_scale=True), use_container_width=True)

# ─── Cumulative Production Chart ───────────────────────────────────────────────
st.markdown("---")
st.markdown("**Cumulative Production Forecast**")

fig_cum = go.Figure()
for name, m in models.items():
    t_cum = np.arange(m['t_aban'])
    if name == 'Exponential':
        cum = cumulative_exp(t_cum, m['qi'], m['di']) / 1000
    elif name == 'Hyperbolic':
        cum = cumulative_hyp(t_cum, m['qi'], m['di'], m['b']) / 1000
    else:
        cum = cumulative_har(t_cum, m['qi'], m['di']) / 1000
    
    fig_cum.add_trace(go.Scatter(
        x=t_cum, y=cum, mode='lines', name=f"{name} ({m['eur_mbbl']:.0f} Mbbl EUR)",
        line=dict(color=m['color'], width=2, dash=m['dash']),
        fill='tozeroy', fillcolor=m['color'].replace(')', ', 0.05)').replace('rgb', 'rgba').replace('#', 'rgba(').replace('60a5fa', '96, 165, 250').replace('34d399', '52, 211, 153').replace('fb923c', '251, 146, 60')
    ))

fig_cum.update_layout(
    template="plotly_dark", paper_bgcolor='#111827', plot_bgcolor='#0d1117',
    font=dict(family='IBM Plex Mono', size=11, color='#9ca3af'),
    legend=dict(bgcolor='#1f2937', bordercolor='#374151', borderwidth=1,
                orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    xaxis=dict(title='Time (months)', gridcolor='#1f2937'),
    yaxis=dict(title='Cumulative Production (Mbbl)', gridcolor='#1f2937'),
    margin=dict(t=60, b=60, l=80, r=20), height=320
)
st.plotly_chart(fig_cum, use_container_width=True)

# ─── Results Table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Model Parameters & Fit Quality**")

if models:
    rows = []
    for name, m in models.items():
        rows.append({
            'Model': name,
            'qi (bopd)': f"{m['qi']:,.0f}",
            'di (month⁻¹)': f"{m['di']:.4f}",
            'b factor': f"{m['b']:.3f}",
            'R²': f"{m['r2']:.5f}",
            'EUR (Mbbl)': f"{m['eur_mbbl']:,.1f}",
            'Revenue (USD MM)': f"${m['eur_mbbl'] * oil_price / 1000:,.2f}",
            'Life (months)': f"{m['t_aban']}"
        })
    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

# ─── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
col_dl1, col_dl2 = st.columns([1, 3])

with col_dl1:
    if models:
        export_rows = []
        for name, m in models.items():
            for t_val in t_forecast:
                if name == 'Exponential':
                    q_val = arps_exponential(t_val, m['qi'], m['di'])
                elif name == 'Hyperbolic':
                    q_val = arps_hyperbolic(t_val, m['qi'], m['di'], m['b'])
                else:
                    q_val = arps_harmonic(t_val, m['qi'], m['di'])
                export_rows.append({'Model': name, 'Month': int(t_val), 'Rate_bopd': round(q_val, 1)})
        
        df_export = pd.DataFrame(export_rows)
        csv_bytes = df_export.to_csv(index=False).encode()
        st.download_button("⬇ Export Forecast CSV", csv_bytes, "dca_forecast.csv", "text/csv")

st.markdown("---")
st.caption("Built with petropt · Arps (1945) · Streamlit · Plotly | Portfolio by [Juang Bhakti H]")
