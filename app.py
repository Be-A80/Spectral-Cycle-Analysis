import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(
    page_title="High-Precision Hurst-Ehlers",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 High-Precision Hurst-Ehlers Zyklus-Modell")
st.caption("Mit exponentieller Gegenwarts-Gewichtung (WLS), Multi-Peak-Spektrum und adaptivem Trend-Fitting.")

# US-kompatible Datenquelle (Kraken)
@st.cache_resource
def get_exchange():
    return ccxt.kraken({'enableRateLimit': True})

exchange = get_exchange()

TF_CONFIG = {
    "15m": {"fetch_tf": "15m", "resample": None, "limit": 720},
    "1h":  {"fetch_tf": "1h",  "resample": None, "limit": 720},
    "4h":  {"fetch_tf": "4h",  "resample": None, "limit": 720},
    "1d":  {"fetch_tf": "1d",  "resample": None, "limit": 720},
    "2d":  {"fetch_tf": "1d",  "resample": "2D", "limit": 720},
    "3d":  {"fetch_tf": "1d",  "resample": "3D", "limit": 720},
    "4d":  {"fetch_tf": "1d",  "resample": "4D", "limit": 720},
    "8d":  {"fetch_tf": "1d",  "resample": "8D", "limit": 720},
    "1w":  {"fetch_tf": "1w",  "resample": None, "limit": 500}
}

# ----------------- Sidebar / Parameter -----------------
with st.sidebar:
    st.header("⚙️ Markt & Zeiteinheit")
    market_list = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "NEAR"]
    selected_coin = st.selectbox("Markt / Asset", options=market_list, index=0)
    
    tf_options = list(TF_CONFIG.keys())
    tf_choice = st.selectbox("Timeframe", options=tf_options, index=tf_options.index("1d"))
    lookback_bars = st.slider(
        "Historischer Analysebereich (Kerzen)",
        min_value=80, max_value=500, value=220, step=20
    )
    
    st.markdown("---")
    st.subheader("🎯 Präzisions-Tuning")
    recency_focus = st.slider(
        "Gegenwarts-Fokus (WLS-Gewichtung)", 
        min_value=0.0, max_value=5.0, value=2.5, step=0.5
    )
    trend_degree = st.selectbox(
        "Basis-Trend Modell", 
        options=[1, 2, 3], 
        index=1,
        format_func=lambda x: "Linear (Gerade)" if x == 1 else f"Adaptives Polynom ({x}. Grades)"
    )
    anchor_to_last = st.checkbox("Projektion exakt am aktuellen Kurs verankern", value=True)
    
    st.markdown("---")
    st.subheader("📡 Ehlers & Hurst Parameter")
    min_period = st.number_input("Min. Zykluslänge (Kerzen)", min_value=4, max_value=30, value=7, step=1)
    max_period = st.number_input("Max. Zykluslänge (Kerzen)", min_value=25, max_value=150, value=85, step=5)
    num_components = st.slider("Anzahl kombinierter Zyklen", min_value=2, max_value=8, value=4, step=1)
    forecast_bars = st.slider("Projektion in Zukunft (Kerzen)", min_value=10, max_value=100, value=30, step=5)
    
    st.button("🔄 Neu berechnen", use_container_width=True)

# ----------------- Daten abrufen -----------------
def fetch_candles(coin, tf_key):
    cfg = TF_CONFIG[tf_key]
    symbol = f"{coin}/USD"
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=cfg["fetch_tf"], limit=cfg["limit"])
    if not ohlcv or len(ohlcv) < 50:
        return None
    
    cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.DataFrame(ohlcv, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
    return df

# ----------------- 1. Multi-Peak Ehlers Spektralanalyse -----------------
def ehlers_multi_peak_spectrum(prices, min_p, max_p, num_comp):
    detrended = np.diff(np.log(prices))
    n = len(detrended)
    
    periods = np.arange(min_p, max_p + 1)
    scores = []
    
    for p in periods:
        if n < 3 * p:
            scores.append(0.0)
            continue
        seg1 = detrended[-p:]
        seg2 = detrended[-2*p : -p]
        seg3 = detrended[-3*p : -2*p]
        
        r1 = 0.0
        if np.std(seg1) > 1e-8 and np.std(seg2) > 1e-8:
            r1 = np.corrcoef(seg1, seg2)[0, 1]
            
        r2 = 0.0
        if np.std(seg2) > 1e-8 and np.std(seg3) > 1e-8:
            r2 = np.corrcoef(seg2, seg3)[0, 1]
        
        combined_r = max(0.0, 0.7 * r1 + 0.3 * r2)
        scores.append(combined_r)
        
    score_arr = np.array(scores)
    peaks, _ = find_peaks(score_arr, distance=3)
    
    if len(peaks) > 0:
        sorted_peaks = peaks[np.argsort(score_arr[peaks])[::-1]]
        selected_p = [float(periods[idx]) for idx in sorted_peaks[:num_comp]]
    else:
        best_idx = int(np.argmax(score_arr))
        selected_p = [float(periods[best_idx])]
        
    dom_period = int(selected_p[0])
    dom_conf = float(np.max(score_arr)) * 100.0 if len(score_arr) > 0 else 0.0
    
    while len(selected_p) < num_comp:
        candidate = max(4.0, round(selected_p[0] / (len(selected_p) + 1), 1))
        if candidate not in selected_p:
            selected_p.append(candidate)
        else:
            break
            
    return dom_period, dom_conf, selected_p, periods, score_arr

# ----------------- 2. Gewichtetes Simultanes Fitting (WLS) -----------------
def fit_high_precision_model(prices, cycle_periods, forecast_n, weight_strength, poly_deg, anchor_last):
    N = len(prices)
    x = np.arange(N)
    total_len = N + forecast_n
    x_full = np.arange(total_len)
    
    if weight_strength > 0:
        weights = np.exp(np.linspace(-weight_strength, 0, N))
    else:
        weights = np.ones(N)
        
    poly_coefs = np.polyfit(x, prices, deg=poly_deg, w=weights)
    trend_hist = np.polyval(poly_coefs, x)
    
    last_val = trend_hist[-1]
    last_slope = trend_hist[-1] - trend_hist[-2]
    trend_future = np.array([last_val + last_slope * (k + 1) * 0.7 for k in range(forecast_n)])
    trend_full = np.concatenate([trend_hist, trend_future])
    
    detrended = prices - trend_hist
    
    basis_cols = []
    for p in cycle_periods:
        w = 2.0 * np.pi / p
        basis_cols.append(np.cos(w * x))
        basis_cols.append(np.sin(w * x))
        
    A_mat = np.column_stack(basis_cols)
    
    W_sqrt = np.sqrt(weights)[:, None]
    A_weighted = A_mat * W_sqrt
    y_weighted = detrended * np.sqrt(weights)
    
    coefs, _, _, _ = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
    
    composite_wave = np.zeros(total_len)
    wave_dict = {}
    
    for idx, p in enumerate(cycle_periods):
        c_cos = coefs[2 * idx]
        c_sin = coefs[2 * idx + 1]
        amp = float(np.sqrt(c_cos**2 + c_sin**2))
        phase = float(np.arctan2(c_sin, c_cos))
        
        w = 2.0 * np.pi / p
        wave_full = amp * np.cos(w * x_full - phase)
        
        if idx == 0:
            label = f"⭐ Hauptzyklus ({p:.1f} Bars)"
        else:
            label = f"Zyklus #{idx+1} ({p:.1f} Bars)"
            
        wave_dict[label] = wave_full
        composite_wave += wave_full
        
    projected_price = trend_full + composite_wave
    
    if anchor_last:
        gap = prices[-1] - projected_price[N - 1]
        decay = np.exp(-np.arange(forecast_n) / (cycle_periods[0] * 0.5))
        projected_price[N:] += gap * decay
        projected_price[:N] += gap * np.exp(-np.arange(N)[::-1] / 15.0)
        
    recent_len = min(50, N)
    corr_matrix = np.corrcoef(prices[-recent_len:], projected_price[N - recent_len : N])
    fit_r2 = corr_matrix[0, 1] * 100.0
    
    return projected_price, wave_dict, max(0.0, float(fit_r2))

# ----------------- Hauptausführung -----------------
with st.spinner(f"Berechne Zyklusmodell für {selected_coin}..."):
    df_raw = fetch_candles(selected_coin, tf_choice)

if df_raw is None or len(df_raw) < lookback_bars:
    st.error("Nicht genügend Kurshistorie verfügbar.")
    st.stop()

df_analysis = df_raw.iloc[-lookback_bars:].copy()
prices = df_analysis['Close'].values
N = len(prices)

dom_period, dom_conf, active_periods, spec_periods, spec_scores = ehlers_multi_peak_spectrum(
    prices, min_period, max_period, num_components
)

projected_price, wave_dict, recent_fit_quality = fit_high_precision_model(
    prices, active_periods, forecast_bars, recency_focus, trend_degree, anchor_to_last
)

time_step = df_analysis.index[-1] - df_analysis.index[-2]
future_times = [df_analysis.index[-1] + time_step * (k + 1) for k in range(forecast_bars)]
all_times = list(df_analysis.index) + future_times

future_curve = projected_price[N:]
peaks, _ = find_peaks(future_curve, distance=2)
troughs, _ = find_peaks(-future_curve, distance=2)

turning_points = []
for p in peaks:
    turning_points.append({
        "Typ": "▼ Zyklus-HOCH (Top)",
        "Zeitpunkt": future_times[p].strftime("%d.%m.%Y %H:%M"),
        "Kerzen entfernt": int(p + 1),
        "Anzeige": f"+{p + 1} Kerzen"
    })
for t in troughs:
    turning_points.append({
        "Typ": "▲ Zyklus-TIEF (Boden)",
        "Zeitpunkt": future_times[t].strftime("%d.%m.%Y %H:%M"),
        "Kerzen entfernt": int(t + 1),
        "Anzeige": f"+{t + 1} Kerzen"
    })

# ----------------- Dashboard Metriken -----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset / Timeframe", f"{selected_coin} / {tf_choice}")
c2.metric("🎯 Dominanter Ehlers-Zyklus", f"{dom_period} Kerzen", f"Konfidenz: {dom_conf:.1f}%")
c3.metric("Gegenwarts-Fit (Letzte 50 Bars)", f"{recent_fit_quality:.1f} %", "Korrelation Kurs vs. Modell")
cycles_str = ", ".join([f"{int(p)}" for p in active_periods]) + " Bars"
c4.metric("Aktive Modell-Zyklen", cycles_str)

st.markdown("---")

# ----------------- Hauptchart -----------------
st.subheader("📈 Realkurs & Gewichtete Hurst-Ehlers Summenkurve")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=df_analysis['Close'],
    mode='lines',
    name='Realer Kurs (Close)',
    line=dict(color='#d1d4dc', width=1.6)
))

fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=projected_price[:N],
    mode='lines',
    name='Gewichteter Zyklus-Fit (WLS)',
    line=dict(color='#3b82f6', width=2, dash='dot')
))

fig.add_trace(go.Scatter(
    x=future_times,
    y=projected_price[N:],
    mode='lines',
    name='🔮 Präzisions-Projektion',
    line=dict(color='#089981', width=3)
))

if len(peaks) > 0:
    fig.add_trace(go.Scatter(
        x=[future_times[p] for p in peaks],
        y=[future_curve[p] for p in peaks],
        mode='markers',
        name='Projiziertes Top ▼',
        marker=dict(color='#f23645', size=10, symbol='triangle-down')
    ))

if len(troughs) > 0:
    fig.add_trace(go.Scatter(
        x=[future_times[t] for t in troughs],
        y=[future_curve[t] for t in troughs],
        mode='markers',
        name='Projizierter Boden ▲',
        marker=dict(color='#089981', size=10, symbol='triangle-up')
    ))

fig.add_vline(
    x=df_analysis.index[-1].timestamp() * 1000,
    line_width=1.5,
    line_dash="dash",
    line_color="#f23645"
)

fig.update_layout(
    template="plotly_dark",
    height=560,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ----------------- Detail-Tabs -----------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Zukünftige Wendepunkte",
    "🌊 Einzelne Zykluskomponenten",
    "📡 Ehlers Multi-Peak Spektrum"
])

with tab1:
    st.markdown("##### Chronologische Liste der nächsten Zyklus-Wendepunkte")
    if turning_points:
        df_tp = pd.DataFrame(turning_points).sort_values(by="Kerzen entfernt")
        df_tp = df_tp.drop(columns=["Kerzen entfernt"])
        st.dataframe(df_tp, use_container_width=True, hide_index=True)
    else:
        st.info("Keine klaren Wendepunkte im gewählten Horizont erkannt.")

with tab2:
    st.markdown("##### Isolierte Zykluswellen (Simultanes WLS-Fitting)")
    fig_waves = go.Figure()
    for w_name, w_vals in wave_dict.items():
        fig_waves.add_trace(go.Scatter(x=all_times, y=w_vals, mode='lines', name=w_name))
    fig_waves.add_vline(
        x=df_analysis.index[-1].timestamp() * 1000,
        line_dash="dash",
        line_color="#f23645"
    )
    fig_waves.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_waves, use_container_width=True)

with tab3:
    st.markdown("##### Ehlers Autokorrelations-Spektrum (Genutzte Peaks grün markiert)")
    spec_df = pd.DataFrame({"Periode": spec_periods, "Score": spec_scores * 100.0})
    active_int = [int(round(p)) for p in active_periods]
    bar_colors = ["#089981" if p in active_int else "#2a2e39" for p in spec_df["Periode"]]
    
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Bar(
        x=spec_df["Periode"],
        y=spec_df["Score"],
        marker_color=bar_colors
    ))
    fig_spec.update_layout(
        template="plotly_dark",
        height=380,
        xaxis_title="Zykluslänge (in Kerzen)",
        yaxis_title="Zyklus-Stärke (%)",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_spec, use_container_width=True)
