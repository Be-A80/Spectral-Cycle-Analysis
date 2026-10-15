import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone, date, timedelta

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(page_title="Hurst Zyklus-Summenkurve", page_icon="🌊", layout="wide")

st.title("🌊 Zykluskomponenten & Composite-Summenkurve (Hurst-Modell)")
st.caption("Ermittelt automatisch dominante Zyklen mittels FFT-Spektralanalyse und projiziert die Summenwelle in die Zukunft.")

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
    
    st.markdown("---")
    st.subheader("🔍 Analyse-Zeitraum & Filter")
    lookback_bars = st.slider("Historischer Analysebereich (Kerzen)", min_value=60, max_value=500, value=250, step=10)
    
    min_cycle = st.number_input("Minimale Zykluslänge (Kerzen)", min_value=3, max_value=30, value=8, step=1)
    max_cycle = st.number_input("Maximale Zykluslänge (Kerzen)", min_value=30, max_value=200, value=90, step=5)
    
    num_cycles = st.slider("Anzahl dominanter Zyklen für Summe", min_value=1, max_value=8, value=3, step=1)
    
    st.markdown("---")
    st.subheader("🔮 Zukunftsprojektion")
    forecast_bars = st.slider("Projektion in die Zukunft (Kerzen)", min_value=10, max_value=120, value=30, step=5)
    include_trend = st.checkbox("Grundtrend in die Summenkurve einrechnen", value=True)
    
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

# ----------------- Zyklusanalyse (FFT Spektralanalyse) -----------------
def analyze_cycles(prices, min_c, max_c, top_k):
    N = len(prices)
    x = np.arange(N)
    
    # 1. Linearen Trend entfernen (Detrending)
    poly = np.polyfit(x, prices, deg=1)
    trend_line = np.polyval(poly, x)
    detrended = prices - trend_line
    
    # Fensterfunktion gegen spektrale Leckage (Tapering)
    window = np.hanning(N)
    windowed = detrended * window
    
    # 2. Fourier-Transformation (FFT)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, d=1.0)
    amplitudes = np.abs(fft_vals) * 2.0 / N
    phases = np.angle(np.fft.rfft(detrended))  # Phase auf Original-Detrended berechnen
    
    # Frequenzen in Perioden (Länge in Kerzen) umrechnen
    periods = np.zeros_like(freqs)
    valid_freq_idx = freqs > 0
    periods[valid_freq_idx] = 1.0 / freqs[valid_freq_idx]
    
    # Nur Zyklen im gewünschten Längenbereich filtern
    in_range = (periods >= min_c) & (periods <= max_c)
    filtered_indices = np.where(in_range)[0]
    
    if len(filtered_indices) == 0:
        return [], trend_line, poly, detrended
    
    # Peaks im Spektrum finden
    amps_in_range = amplitudes[filtered_indices]
    peak_sub_idx, _ = find_peaks(amps_in_range)
    
    if len(peak_sub_idx) > 0:
        candidate_idx = filtered_indices[peak_sub_idx]
    else:
        candidate_idx = filtered_indices
        
    # Nach Amplitude (Stärke) sortieren
    sorted_candidates = candidate_idx[np.argsort(amplitudes[candidate_idx])[::-1]]
    top_indices = sorted_candidates[:top_k]
    
    components = []
    for idx in top_indices:
        p_len = periods[idx]
        freq = freqs[idx]
        amp = amplitudes[idx]
        phase = phases[idx]
        components.append({
            "period": round(float(p_len), 1),
            "freq": float(freq),
            "amp": float(amp),
            "phase": float(phase)
        })
        
    return components, trend_line, poly, detrended

# ----------------- Hauptausführung -----------------
with st.spinner(f"Lade Kursdaten für {selected_coin}..."):
    df_raw = fetch_candles(selected_coin, tf_choice)

if df_raw is None or len(df_raw) < lookback_bars:
    st.error("Nicht genügend Kurshistorie verfügbar.")
    st.stop()

# Analysefenster auswählen
df_analysis = df_raw.iloc[-lookback_bars:].copy()
prices = df_analysis['Close'].values
N = len(prices)

components, hist_trend, poly, detrended = analyze_cycles(prices, min_cycle, max_cycle, num_cycles)

if not components:
    st.warning("Im gewählten Frequenzband konnten keine signifikanten Zyklen identifiziert werden.")
    st.stop()

# ----------------- Zukunft & Summenkurve berechnen -----------------
total_len = N + forecast_bars
x_full = np.arange(total_len)

# Zukünftige Zeitstempel generieren
time_step = df_analysis.index[-1] - df_analysis.index[-2]
future_times = [df_analysis.index[-1] + time_step * (k + 1) for k in range(forecast_bars)]
all_times = list(df_analysis.index) + future_times

# Trendlinie über Vergangenheit und Zukunft
trend_full = np.polyval(poly, x_full)

# Einzelne Sinuswellen und Summenwelle berechnen
composite_wave = np.zeros(total_len)
wave_dict = {}

for c in components:
    w = c["freq"] * 2 * np.pi
    # Welle: y(t) = A * cos(w * t + phase)
    single_wave = c["amp"] * np.cos(w * x_full + c["phase"])
    wave_dict[f"Zyklus {c['period']} Bars (Stärke {c['amp']:.1f})"] = single_wave
    composite_wave += single_wave

# Finale Projektionskurve
if include_trend:
    projected_price = trend_full + composite_wave
else:
    # Ohne Trend: um den letzten Kurs zentrieren
    last_price = prices[-1]
    projected_price = (composite_wave - composite_wave[N - 1]) + last_price

# ----------------- Wendepunkte in der Zukunft finden -----------------
future_comp = projected_price[N:]
f_peaks, _ = find_peaks(future_comp)
f_troughs, _ = find_peaks(-future_comp)

turning_points = []
for p in f_peaks:
    tp_time = future_times[p]
    turning_points.append({
        "Typ": "▼ Zyklus-HOCH (Top)",
        "Zeitpunkt": tp_time.strftime("%d.%m.%Y %H:%M"),
        "Kerzen in Zukunft": f"+{p + 1}"
    })
for t in f_troughs:
    tp_time = future_times[t]
    turning_points.append({
        "Typ": "▲ Zyklus-TIEF (Boden)",
        "Zeitpunkt": tp_time.strftime("%d.%m.%Y %H:%M"),
        "Kerzen in Zukunft": f"+{t + 1}"
    })

# ----------------- UI Dashboard -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Asset / Markt", f"{selected_coin} / {tf_choice}")
col2.metric("Dominanter Zyklus #1", f"{components[0]['period']} Kerzen")
col3.metric("Extrahierte Zyklen", f"{len(components)}")
col4.metric("Projektionshorizont", f"+{forecast_bars} Kerzen")

st.markdown("---")

# Interaktiver Chart: Kurs + Summenkurve
st.subheader("📈 Kursverlauf & Projizierte Hurst-Summenkurve")

fig = go.Figure()

# Reale Kerzenkurse (Vergangenheit)
fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=df_analysis['Close'],
    mode='lines',
    name='Realer Kurs (Close)',
    line=dict(color='#d1d4dc', width=1.5)
))

# Summenkurve Vergangenheit (Fit)
fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=projected_price[:N],
    mode='lines',
    name='Summenkurve (Historischer Fit)',
    line=dict(color='#3b82f6', width=2, dash='dot')
))

# Summenkurve Zukunft (Projektion)
fig.add_trace(go.Scatter(
    x=future_times,
    y=projected_price[N:],
    mode='lines',
    name='🔮 Summenkurven-Projektion (Zukunft)',
    line=dict(color='#089981', width=3)
))

# Vertikale Trennlinie JETZT
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
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# Detail-Reiter: Einzelwellen, Wendepunkte und Spektrum
tab1, tab2, tab3 = st.tabs(["🌊 Einzelne Zykluskomponenten", "🎯 Zukünftige Wendepunkte", "📊 Zyklus-Spektrum"])

with tab1:
    st.markdown("##### Isolierte Sinuswellen (Die Bausteine der Summenkurve)")
    st.caption("Jeder Markt besteht nach Hurst aus der Überlagerung mehrerer unabhängiger Zyklen:")
    
    fig_waves = go.Figure()
    for wave_name, wave_vals in wave_dict.items():
        fig_waves.add_trace(go.Scatter(
            x=all_times,
            y=wave_vals,
            mode='lines',
            name=wave_name
        ))
    
    fig_waves.add_vline(x=df_analysis.index[-1].timestamp() * 1000, line_dash="dash", line_color="#f23645")
    fig_waves.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_waves, use_container_width=True)

with tab2:
    st.markdown("##### Errechnete Wendepunkte auf der Summenkurve")
    st.caption("Zeitpunkte, an denen die überlagerten Zyklen gemeinsam ein Top oder Bottom ausbilden:")
    if turning_points:
        df_tp = pd.DataFrame(turning_points).sort_values(by="Kerzen in Zukunft")
        st.dataframe(df_tp, use_container_width=True, hide_index=True)
    else:
        st.info("Im gewählten Vorhersagezeitraum wurden keine Extrempunkte erkannt.")

with tab3:
    st.markdown("##### Rangliste der stärksten identifizierten Zyklen")
    comp_df = pd.DataFrame(components)[["period", "amp"]].rename(columns={
        "period": "Zykluslänge (Kerzen)",
        "amp": "Amplitude / Stärke"
    })
    st.dataframe(
        comp_df,
        column_config={
            "Zykluslänge (Kerzen)": st.column_config.NumberColumn(format="%.1f"),
            "Amplitude / Stärke": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=float(comp_df["Amplitude / Stärke"].max() * 1.1))
        },
        use_container_width=True,
        hide_index=True
    )
