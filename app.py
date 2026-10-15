import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone, date, timedelta

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(page_title="Ehlers Dominant Cycle & Summenkurve", page_icon="📡", layout="wide")

st.title("📡 Ehlers Dominant Cycle Finder & Composite-Projektion")
st.caption("Objektive Erkennung des dominanten Zyklus mittels Ehlers-Autokorrelation mit anschließender Projektion.")

# US-kompatible Datenquelle (Kraken)
@st.cache_resource
def get_exchange():
    return ccxt.kraken({
        'enableRateLimit': True,
    })

exchange = get_exchange()

TF_CONFIG = {
    "1h":  {"fetch_tf": "1h", "resample": None,  "limit": 720},
    "4h":  {"fetch_tf": "4h", "resample": None,  "limit": 720},
    "1d":  {"fetch_tf": "1d", "resample": None,  "limit": 720},
    "2d":  {"fetch_tf": "1d", "resample": "2D",  "limit": 720},
    "3d":  {"fetch_tf": "1d", "resample": "3D",  "limit": 720},
    "4d":  {"fetch_tf": "1d", "resample": "4D",  "limit": 720},
    "1w":  {"fetch_tf": "1w", "resample": None,  "limit": 500}
}

# ----------------- Sidebar / Parameter -----------------
with st.sidebar:
    st.header("⚙️ Markt & Daten")
    market_list = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "NEAR"]
    selected_coin = st.selectbox("Markt / Asset", options=market_list, index=0)
    
    tf_options = list(TF_CONFIG.keys())
    tf_choice = st.selectbox("Timeframe", options=tf_options, index=tf_options.index("1d"))
    lookback_bars = st.slider("Historischer Analysebereich (Kerzen)", min_value=80, max_value=600, value=300, step=20)
    
    st.markdown("---")
    st.subheader("📡 Ehlers Cycle Scanner")
    min_period = st.number_input("Minimale Periode (Kerzen)", min_value=5, max_value=30, value=8, step=1)
    max_period = st.number_input("Maximale Periode (Kerzen)", min_value=20, max_value=150, value=70, step=5)
    
    st.markdown("---")
    st.subheader("🔮 Projektion")
    forecast_bars = st.slider("Projektionshorizont (Kerzen)", min_value=10, max_value=100, value=30, step=5)
    include_trend = st.checkbox("Grundtrend in Summenkurve einrechnen", value=True)
    
    st.button("🔄 Analyse ausführen", use_container_width=True)

# ----------------- Daten abrufen -----------------
def fetch_candles(coin, tf_choice):
    cfg = TF_CONFIG[tf_choice]
    symbol = f"{coin}/USD"
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=cfg["fetch_tf"], limit=cfg["limit"])
    if not ohlcv or len(ohlcv) < 50:
        return None
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
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

# ----------------- Ehlers Autokorrelations-Algorithmus -----------------
def ehlers_autocorrelation_periodogram(prices, min_p, max_p):
    """
    Berechnet das Ehlers-Autokorrelations-Spektrum.
    Ermittelt die Korrelation zwischen Zeitfenstern zur Erkennung des dominanten Zyklus.
    """
    # 1. Detrending über Log-Returns (stationäre Zeitreihe)
    detrended = np.diff(np.log(prices))
    n = len(detrended)
    
    periods = np.arange(min_p, max_p + 1)
    correlations = []
    
    for p in periods:
        if n < 2 * p:
            correlations.append(0.0)
            continue
        
        # Die letzten p Kerzen mit den vorangegangenen p Kerzen korrelieren
        x = detrended[-p:]
        y = detrended[-2 * p : -p]
        
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x > 1e-8 and std_y > 1e-8:
            r = np.corrcoef(x, y)[0, 1]
            correlations.append(max(0.0, r))  # Negative Korrelationen auf 0 kappen
        else:
            correlations.append(0.0)
            
    corr_arr = np.array(correlations)
    
    # Höchsten Peak als dominanten Zyklus identifizieren
    if len(corr_arr) == 0 or np.all(corr_arr == 0):
        return int(min_p), 0.0, periods, corr_arr
        
    best_idx = np.argmax(corr_arr)
    dom_period = int(periods[best_idx])
    confidence = float(corr_arr[best_idx]) * 100.0  # In Prozent
    
    return dom_period, confidence, periods, corr_arr

# ----------------- Multi-Harmonische Rekonstruktion -----------------
def fit_cyclical_model(prices, dom_p, forecast_n):
    """
    Passt eine Sinusschwingung des dominanten Zyklus sowie seiner Oberschwingung (Harmonische)
    an die Kurshistorie an und projiziert sie in die Zukunft.
    """
    N = len(prices)
    x = np.arange(N)
    
    # 1. Linearen Trend berechnen
    poly = np.polyfit(x, prices, deg=1)
    trend_hist = np.polyval(poly, x)
    detrended = prices - trend_hist
    
    # 2. Fourier-Koeffizienten für die dominante Frequenz & Harmonische berechnen
    # Grundfrequenz w1 = 2*pi / dom_p, erste Oberschwingung w2 = 2*pi / (dom_p / 2)
    periods_to_fit = [dom_p, max(4.0, dom_p / 2.0)]
    
    total_len = N + forecast_n
    x_full = np.arange(total_len)
    composite_wave = np.zeros(total_len)
    wave_details = {}
    
    for idx, p in enumerate(periods_to_fit):
        w = 2.0 * np.pi / p
        sin_basis = np.sin(w * x)
        cos_basis = np.cos(w * x)
        
        # Least-Squares Fit für Amplitude und Phase
        A_mat = np.column_stack([cos_basis, sin_basis])
        coefs, _, _, _ = np.linalg.lstsq(A_mat, detrended, rcond=None)
        
        amp = np.sqrt(coefs[0]**2 + coefs[1]**2)
        phase = np.arctan2(coefs[1], coefs[0])
        
        # Über gesamte Zeitspanne projizieren
        fit_wave = amp * np.cos(w * x_full - phase)
        label = f"Dominanter Hauptzyklus ({p:.1f} Bars)" if idx == 0 else f"Oberschwingung / Harmonische ({p:.1f} Bars)"
        wave_details[label] = fit_wave
        composite_wave += fit_wave
        
    trend_full = np.polyval(poly, x_full)
    return composite_wave, trend_full, wave_details

# ----------------- Hauptausführung -----------------
with st.spinner(f"Lade Kursdaten für {selected_coin}..."):
    df_raw = fetch_candles(selected_coin, tf_choice)

if df_raw is None or len(df_raw) < lookback_bars:
    st.error("Nicht genügend Kurshistorie verfügbar.")
    st.stop()

df_analysis = df_raw.iloc[-lookback_bars:].copy()
prices = df_analysis['Close'].values
N = len(prices)

# 1. Ehlers Dominant Cycle ermitteln
dom_period, confidence, all_periods, all_corrs = ehlers_autocorrelation_periodogram(prices, min_period, max_period)

# 2. Zyklische Summenkurve berechnen
composite_wave, trend_full, wave_details = fit_cyclical_model(prices, dom_period, forecast_bars)

time_step = df_analysis.index[-1] - df_analysis.index[-2]
future_times = [df_analysis.index[-1] + time_step * (k + 1) for k in range(forecast_bars)]
all_times = list(df_analysis.index) + future_times

if include_trend:
    projected_price = trend_full + composite_wave
else:
    last_price = prices[-1]
    projected_price = (composite_wave - composite_wave[N - 1]) + last_price

# 3. Zukünftige Wendepunkte extrahieren
future_curve = projected_price[N:]
peaks, _ = find_peaks(future_curve)
troughs, _ = find_peaks(-future_curve)

future_events = []
for p in peaks:
    future_events.append({
        "Typ": "▼ Zyklus-HOCH (Top)",
        "Zeitpunkt": future_times[p].strftime("%d.%m.%Y %H:%M"),
        "In Zukunft": f"+{p + 1} Kerzen"
    })
for t in troughs:
    future_events.append({
        "Typ": "▲ Zyklus-TIEF (Boden)",
        "Zeitpunkt": future_times[t].strftime("%d.%m.%Y %H:%M"),
        "In Zukunft": f"+{t + 1} Kerzen"
    })

# ----------------- Dashboard Metriken -----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset / Timeframe", f"{selected_coin} / {tf_choice}")
c2.metric("🎯 Dominanter Zyklus (Ehlers)", f"{dom_period} Kerzen")
c3.metric("Signalstärke / Konfidenz", f"{confidence:.1f} %")
c4.metric("Projektion", f"+{forecast_bars} Kerzen")

st.markdown("---")

# Interaktiver Chart: Kurs + Summenkurve
st.subheader("📈 Realkurs & Ehlers-Composite-Summenkurve")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=df_analysis['Close'],
    mode='lines',
    name='Realer Kurs (Close)',
    line=dict(color='#d1d4dc', width=1.5)
))

fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=projected_price[:N],
    mode='lines',
    name='Zyklus-Fit (Historie)',
    line=dict(color='#3b82f6', width=2, dash='dot')
))

fig.add_trace(go.Scatter(
    x=future_times,
    y=projected_price[N:],
    mode='lines',
    name='🔮 Ehlers Zukunftsprojektion',
    line=dict(color='#089981', width=3)
))

fig.add_vline(
    x=df_analysis.index[-1].timestamp() * 1000,
    line_width=1.5,
    line_dash="dash",
    line_color="#f23645"
)
fig.add_annotation(
    x=df_analysis.index[-1],
    y=prices[-1],
    text="JETZT",
    showarrow=True,
    arrowhead=1,
    arrowcolor="#f23645",
    font=dict(color="#f23645", size=12)
)

fig.update_layout(
    template="plotly_dark",
    height=530,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# Detail-Tabs
tab1, tab2, tab3 = st.tabs(["📡 Ehlers Periodogramm (Spektrum)", "🎯 Zukünftige Wendepunkte", "🌊 Einzelkomponenten"])

with tab1:
    st.markdown("##### Autokorrelations-Spektrum nach John Ehlers")
    st.caption("Der höchste Peak markiert die Periode mit der stärksten zyklischen Selbstähnlichkeit im Markt:")
    
    spec_df = pd.DataFrame({"Periode": all_periods, "Korrelation": all_corrs * 100.0})
    
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Bar(
        x=spec_df["Periode"],
        y=spec_df["Korrelation"],
        marker_color=["#089981" if p == dom_period else "#2a2e39" for p in spec_df["Periode"]],
        name="Korrelation (%)"
    ))
    fig_spec.update_layout(
        template="plotly_dark",
        height=380,
        xaxis_title="Zykluslänge (in Kerzen)",
        yaxis_title="Autokorrelations-Stärke (%)",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_spec, use_container_width=True)

with tab2:
    st.markdown("##### Berechnete Wendepunkte der Projektionskurve")
    st.caption("Exakte Zeitpunkte, an denen der dominante Zyklus Böden oder Hochs formt:")
    if future_events:
        df_ev = pd.DataFrame(future_events).sort_values(by="In Zukunft")
        st.dataframe(df_ev, use_container_width=True, hide_index=True)
    else:
        st.info("Im gewählten Projektionsbereich wurden keine Wendepunkte gefunden.")

with tab3:
    st.markdown("##### Isolierte Schwingungen")
    st.caption("Der dominante Ehlers-Zyklus und seine erste Harmonische separat dargestellt:")
    fig_w = go.Figure()
    for w_name, w_vals in wave_details.items():
        fig_w.add_trace(go.Scatter(x=all_times, y=w_vals, mode='lines', name=w_name))
    fig_w.add_vline(x=df_analysis.index[-1].timestamp() * 1000, line_dash="dash", line_color="#f23645")
    fig_w.update_layout(
        template="plotly_dark",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_w, use_container_width=True)
