import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(page_title="High-Precision Hurst-Ehlers", page_icon="🎯", layout="wide")

st.title("🎯 High-Precision Hurst-Ehlers Zyklus-Modell")
st.caption("Mit exponentieller Gegenwarts-Gewichtung (WLS), Multi-Peak-Spektrum und adaptivem Trend-Fitting.")

# US-kompatible Datenquelle (Kraken)
@st.cache_resource
def get_exchange():
    return ccxt.kraken({'enableRateLimit': True})

exchange = get_exchange()

TF_CONFIG = {
    "15m": {"fetch_tf": "15m", "resample": None,  "limit": 720},
    "1h":  {"fetch_tf": "1h",  "resample": None,  "limit": 720},
    "4h":  {"fetch_tf": "4h",  "resample": None,  "limit": 720},
    "1d":  {"fetch_tf": "1d",  "resample": None,  "limit": 720},
    "2d":  {"fetch_tf": "1d",  "resample": "2D",  "limit": 720},
    "3d":  {"fetch_tf": "1d",  "resample": "3D",  "limit": 720},
    "4d":  {"fetch_tf": "1d",  "resample": "4D",  "limit": 720},
    "8d":  {"fetch_tf": "1d",  "resample": "8D",  "limit": 720},
    "1w":  {"fetch_tf": "1w",  "resample": None,  "limit": 500}
}

# ----------------- Sidebar / Parameter -----------------
with st.sidebar:
    st.header("⚙️ Markt & Zeiteinheit")
    market_list = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "NEAR"]
    selected_coin = st.selectbox("Markt / Asset", options=market_list, index=0)
    
    tf_options = list(TF_CONFIG.keys())
    tf_choice = st.selectbox("Timeframe", options=tf_options, index=tf_options.index("1d"))
    lookback_bars = st.slider("Historischer Analysebereich (Kerzen)", min_value=80, max_value=500, value=220, step=20)
    
    st.markdown("---")
    st.subheader("🎯 Präzisions-Tuning (Neu)")
    recency_focus = st.slider(
        "Gegenwarts-Fokus (WLS-Gewichtung)", 
        min_value=0.0, max_value=5.0, value=2.5, step=0.5,
        help="Je höher, desto stärker rastet die Kurve auf den jüngsten Hochs und Tiefs am rechten Chartrand ein (0 = alle Kerzen gleich gewichtet)."
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
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
    return df

# ----------------- 1. Multi-Peak Ehlers Spektralanalyse -----------------
def ehlers_multi_peak_spectrum(prices, min_p, max_p, num_comp):
    """
    Berechnet das Ehlers-Autokorrelations-Periodogramm mit Bartels-Konsistenzprüfung
    und extrahiert die echten dominanten Zyklus-Peaks.
    """
    detrended = np.diff(np.log(prices))
    n = len(detrended)
    
    periods = np.arange(min_p, max_p + 1)
    scores = []
    
    for p in periods:
        if n < 3 * p:
            scores.append(0.0)
            continue
        # Korrelation über die letzten beiden Zyklus-Durchläufe (Gegenwarts-stabil)
        seg1 = detrended[-p:]
        seg2 = detrended[-2*p : -p]
        seg3 = detrended[-3*p : -2*p]
        
        r1 = np.corrcoef(seg1, seg2)[0, 1] if (np.std(seg1) > 1e-8 and np.std(seg2) > 1e-8) else 0.0
        r2 = np.corrcoef(seg2, seg3)[0, 1] if (np.std(seg2) > 1e-8 and np.std(seg3) > 1e-8) else 0.0
        
        # Jüngster Durchlauf zählt zu 70 %, vorletzter zu 30 % (Stabilitätsfilter)
        combined_r = max(0.0, 0.7 * r1 + 0.3 * r2)
        scores.append(combined_r)
        
    score_arr = np.array(scores)
    
    # Echte lokale Peaks im Spektrum finden (Mindestabstand 3 Kerzen)
    peaks, _ = find_peaks(score_arr, distance=3)
    
    if len(peaks) > 0:
        sorted_peaks = peaks[np.argsort(score_arr[peaks])[::-1]]
        selected_p = [float(periods[idx]) for idx in sorted_peaks[:num_comp]]
    else:
        best_idx = int(np.argmax(score_arr))
        selected_p = [float(periods[best_idx])]
        
    dom_period = int(selected_p[0])
    dom_conf = float(np.max(score_arr)) * 100.0 if len(score_arr) > 0 else 0.0
    
    # Falls weniger Peaks gefunden als gewünscht, mit Hurst-Harmonischen auffüllen
    while len(selected_p) < num_comp:
        candidate = max(4.0, round(selected_p[0] / (len(selected_p) + 1), 1))
        if candidate not in selected_p:
            selected_p.append(candidate)
        else:
            break
            
    return dom_period, dom_conf, selected_p, periods, score_arr

# ----------------- 2. Gewichtetes Simultanes Fitting (WLS) -----------------
def fit_high_precision_model(prices, cycle_periods, forecast_n, weight_strength, poly_deg, anchor_last):
    """
    Passt alle Zykluswellen und den Grundtrend SIMULTAN mit exponentieller 
    Gegenwarts-Gewichtung (Weighted Least Squares) an den Kurs an.
    """
    N = len(prices)
    x = np.arange(N)
    total_len = N + forecast_n
    x_full = np.arange(total_len)
    
    # Exponentielle Gewichte: Jüngste Kerzen am rechten Rand erhalten das höchste Gewicht
    if weight_strength > 0:
        weights = np.exp(np.linspace(-weight_strength, 0, N))
    else:
        weights = np.ones(N)
        
    # 1. Gewichteter Polynom-Trend (Gedämpft in der Zukunft gegen Überschießen)
    poly_coefs = np.polyfit(x, prices, deg=poly_deg, w=weights)
    trend_hist = np.polyval(poly_coefs, x)
    
    # Für die Zukunft extrapolieren wir die letzte Tangente (verhindert extremes Wegbiegen bei Grad 2/3)
    last_val = trend_hist[-1]
    last_slope = trend_hist[-1] - trend_hist[-2]
    trend_future = np.array([last_val + last_slope * (k + 1) * 0.7 for k in range(forecast_n)])
    trend_full = np.concatenate([trend_hist, trend_future])
    
    detrended = prices - trend_hist
    
    # 2. Simultane Design-Matrix für alle Zyklen aufbauen
    basis_cols = []
    for p in cycle_periods:
        w = 2.0 * np.pi / p
        basis_cols.append(np.cos(w * x))
        basis_cols.append(np.sin(w * x))
        
    A_mat = np.column_stack(basis_cols)
    
    # Gewichtetes Gleichungssystem lösen (WLS)
    W_sqrt = np.sqrt(weights)[:, None]
    A_weighted = A_mat * W_sqrt
    y_weighted = detrended * np.sqrt(weights)
    
    coefs, _, _, _ = np.linalg.lstsq(A_weighted, y_weighted, rcond=None)
    
    # 3. Einzelwellen und Summenwelle konstruieren
    composite_wave = np.zeros(total_len)
    wave_dict = {}
    
    for idx, p in enumerate(cycle_periods):
        c_cos = coefs[2 * idx]
        c_sin = coefs[2 * idx + 1]
        amp = float(np.sqrt(c_cos**2 + c_sin**2))
        phase = float(np.arctan2(c_sin, c_cos))
        
        w = 2.0 * np.pi / p
        wave_full = amp * np.cos(w * x_full - phase)
        
        label = f"⭐ Hauptzyklus ({p:.1f} Bars)" if idx == 0 else
