import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone, date, timedelta

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(page_title="Hurst-Ehlers Composite Cycle Screener", page_icon="🌊", layout="wide")

st.title("🌊 Hurst Composite Cycle Model & Ehlers Dominant Detector")
st.caption("Verbindet die Ehlers-Autokorrelation mit dem Hurst-Spektralmodell zur Vorhersage zukünftiger Zyklus-Wendepunkte.")

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
    "8d":  {"fetch_tf": "1d", "resample": "8D",  "limit": 720},
    "16d": {"fetch_tf": "1d", "resample": "16D", "limit": 720},
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
    st.subheader("📡 Ehlers Autokorrelation")
    min_period = st.number_input("Min. Zykluslänge (Kerzen)", min_value=4, max_value=30, value=8, step=1)
    max_period = st.number_input("Max. Zykluslänge (Kerzen)", min_value=25, max_value=150, value=90, step=5)
    
    st.markdown("---")
    st.subheader("🌊 Hurst-Summenkurve")
    num_harmonics = st.slider("Anzahl zyklischer Komponenten", min_value=2, max_value=6, value=4, step=1,
                              help="Kombiniert den Ehlers-Dominantzyklus mit seinen harmonischen Ober- und Unterwellen nach Hurst.")
    forecast_bars = st.slider("Projektionshorizont in Zukunft (Kerzen)", min_value=10, max_value=120, value=35, step=5)
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

# ----------------- 1. Ehlers Autokorrelations-Algorithmus -----------------
def find_dominant_cycle_ehlers(prices, min_p, max_p):
    """Ermittelt den dominanten Zyklus mittels Ehlers-Autokorrelations-Periodogramm."""
    detrended = np.diff(np.log(prices))
    n = len(detrended)
    
    periods = np.arange(min_p, max_p + 1)
    correlations = []
    
    for p in periods:
        if n < 2 * p:
            correlations.append(0.0)
            continue
        x = detrended[-p:]
        y = detrended[-2 * p : -p]
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x > 1e-8 and std_y > 1e-8:
            r = np.corrcoef(x, y)[0, 1]
            correlations.append(max(0.0, r))
        else:
            correlations.append(0.0)
            
    corr_arr = np.array(correlations)
    if len(corr_arr) == 0 or np.all(corr_arr == 0):
        return int(min_p), 0.0, periods, corr_arr
        
    best_idx = np.argmax(corr_arr)
    dom_period = int(periods[best_idx])
    confidence = float(corr_arr[best_idx]) * 100.0
    
    return dom_period, confidence, periods, corr_arr

# ----------------- 2. Hurst-Modell mit Ehlers-Anker -----------------
def build_hurst_composite_model(prices, dom_period, num_comp, forecast_n):
    """
    Kombiniert den Ehlers-Dominantzyklus mit FFT-abgeleiteten harmonischen Komponenten 
    nach Hurst (Sub- und Oberwellen), passt Phasen und Amplituden an und berechnet 
    die Summenwelle.
    """
    N = len(prices)
    x = np.arange(N)
    
    # 1. Linearen Trend berechnen & entfernen
    poly = np.polyfit(x, prices, deg=1)
    trend_hist = np.polyval(poly, x)
    detrended = prices - trend_hist
    
    # 2. Harmonische Perioden nach Hursts Nominalmodell um den dominanten Ehlers-Zyklus erzeugen
    # Hurst-Prinzip der harmonischen Verhältnisse: T, T/2, T/3, 2T etc.
    candidate_periods = [
        float(dom_period),
        max(3.0, dom_period / 2.0),
        max(3.0, dom_period / 3.0),
        dom_period * 2.0,
        max(4.0, dom_period / 4.0),
        dom_period * 1.5
    ]
    # Auf die gewünschte Anzahl stutzen
    selected_periods = sorted(list(dict.fromkeys(candidate_periods))[:num_comp], reverse=True)
    
    total_len = N + forecast_n
    x_full = np.arange(total_len)
    composite_wave = np.zeros(total_len)
    wave_details = {}
    components_meta = []
    
    # 3. Jede Welle per Least-Squares an die detrendeten Daten anpassen
    for idx, p in enumerate(selected_periods):
        w = 2.0 * np.pi / p
        cos_basis = np.cos(w * x)
        sin_basis = np.sin(w * x)
        
        A_mat = np.column_stack([cos_basis, sin_basis])
        coefs, _, _, _ = np.linalg.lstsq(A_mat, detrended, rcond=None)
        
        amp = float(np.sqrt(coefs[0]**2 + coefs[1]**2))
        phase = float(np.arctan2(coefs[1], coefs[0]))
        
        # Welle über Historie + Zukunft generieren
        fit_wave = amp * np.cos(w * x_full - phase)
        
        if p == dom_period:
            label = f"⭐ Dominanter Ehlers-Zyklus ({p:.1f} Bars)"
        elif p < dom_period:
            label = f"Oberschwingung ({p:.1f} Bars)"
        else:
            label = f"Übergeordneter Makrozyklus ({p:.1f} Bars)"
            
        wave_details[label] = fit_wave
        composite_wave += fit_wave
        
        components_meta.append({
            "period": round(p, 1),
            "amp": round(amp, 2),
            "is_dominant": (p == dom_period),
            "label": label
        })
        
    trend_full = np.polyval(poly, x_full)
    return composite_wave, trend_full, wave_details, components_meta

# ----------------- Hauptausführung -----------------
with st.spinner(f"Lade Kursdaten für {selected_coin}..."):
    df_raw = fetch_candles(selected_coin, tf_choice)

if df_raw is None or len(df_raw) < lookback_bars:
    st.error("Nicht genügend Kurshistorie verfügbar.")
    st.stop()

df_analysis = df_raw.iloc[-lookback_bars:].copy()
prices = df_analysis['Close'].values
N = len(prices)

# 1. Ehlers Dominanten Zyklus berechnen
dom_period, confidence, all_periods, all_corrs = find_dominant_cycle_ehlers(prices, min_period, max_period)

# 2. Hurst Composite Model um den Ehlers-Zyklus aufbauen
composite_wave, trend_full, wave_dict, comp_meta = build_hurst_composite_model(
    prices, dom_period, num_harmonics, forecast_bars
)

# Zeitstempel für Zukunft erzeugen
time_step = df_analysis.index[-1] - df_analysis.index[-2]
future_times = [df_analysis.index[-1] + time_step * (k + 1) for k in range(forecast_bars)]
all_times = list(df_analysis.index) + future_times

# Finale Projektionskurve zusammensetzen
if include_trend:
    projected_price = trend_full + composite_wave
else:
    last_price = prices[-1]
    projected_price = (composite_wave - composite_wave[N - 1]) + last_price

# 3. Zukünftige Wendepunkte auf der Summenkurve finden
future_curve = projected_price[N:]
peaks, _ = find_peaks(future_curve)
troughs, _ = find_peaks(-future_curve)

turning_points = []
for p in peaks:
    turning_points.append({
        "Typ": "▼ Zyklus-HOCH (Top)",
        "Zeitpunkt": future_times[p].strftime("%d.%m.%Y %H:%M"),
        "Kerzen in Zukunft": f"+{p + 1}"
    })
for t in troughs:
    turning_points.append({
        "Typ": "▲ Zyklus-TIEF (Boden)",
        "Zeitpunkt": future_times[t].strftime("%d.%m.%Y %H:%M"),
        "Kerzen in Zukunft": f"+{t + 1}"
    })

# ----------------- Dashboard Metriken -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Asset / Markt", f"{selected_coin} / {tf_choice}")
col2.metric("🎯 Dominanter Ehlers-Zyklus", f"{dom_period} Kerzen", f"{confidence:.1f}% Konfidenz")
col3.metric("Hurst-Komponenten", f"{len(comp_meta)} Wellen")
col4.metric("Projektionshorizont", f"+{forecast_bars} Kerzen")

st.markdown("---")

# ----------------- Hauptchart: Kurs + Hurst-Summenkurve -----------------
st.subheader("📈 Realkurs & Hurst-Summenkurve (Ehlers-basiert)")

fig = go.Figure()

# Reale Kerzenkurse (Vergangenheit)
fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=df_analysis['Close'],
    mode='lines',
    name='Realer Kurs (Close)',
    line=dict(color='#d1d4dc', width=1.5)
))

# Summenkurve Vergangenheit (Historischer Fit)
fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=projected_price[:N],
    mode='lines',
    name='Hurst-Summenkurve (Fit)',
    line=dict(color='#3b82f6', width=2, dash='dot')
))

# Summenkurve Zukunft (Projektion)
fig.add_trace(go.Scatter(
    x=future_times,
    y=projected_price[N:],
    mode='lines',
    name='🔮 Zukunftsprojektion (Composite)',
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

# ----------------- Detail-Tabs -----------------
tab1, tab2, tab3 = st.tabs([
    "🌊 Einzelne Zykluskomponenten",
    "🎯 Zukünftige Wendepunkte",
    "📡 Ehlers Periodogramm & Spektrum"
])

with tab1:
    st.markdown("##### Isolierte Schwingungswellen (Die Hurst-Bausteine)")
    st.caption("Zeigt den dominanten Ehlers-Zyklus und seine harmonischen Ober- und Unterwellen separat an:")
    
    fig_waves = go.Figure()
    for wave_name, wave_vals in wave_dict.items():
        is_dom = "Dominanter" in wave_name
        fig_waves.add_trace(go.Scatter(
            x=all_times,
            y=wave_vals,
            mode='lines',
            name=wave_name,
            line=dict(width=2.5 if is_dom else 1.2)
        ))
    
    fig_waves.add_vline(x=df_analysis.index[-1].timestamp() * 1000, line_dash="dash", line_color="#f23645")
    fig_waves.update_layout(
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_waves, use_container_width=True)

with tab2:
    st.markdown("##### Errechnete Wendepunkte der Summenkurve")
    st.caption("Konkrete Zeitpunkte in der Zukunft, an denen die zusammengesetzte Hurst-Welle Spitzen (Tops) oder Täler (Böden) formt:")
    if turning_points:
        df_tp = pd.DataFrame(turning_points).sort_values(by="Kerzen in Zukunft")
        st.dataframe(df_tp, use_container_width=True, hide_index=True)
    else:
        st.info("Im gewählten Vorhersagezeitraum wurden keine Extrempunkte erkannt.")

with tab3:
    st.markdown("##### Ehlers Autokorrelations-Spektrum")
    st.caption(f"Der grüne Balken hebt den erkannten Haupttaktgeber ({dom_period} Kerzen) hervor:")
    
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
