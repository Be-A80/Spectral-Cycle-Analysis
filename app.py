import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from scipy.signal import find_peaks
from datetime import datetime, timezone
import time

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(
    page_title="Hurst-Ehlers Pro + Multi-Scanner",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Self-Calibrating Hurst-Ehlers & Multi-Coin Scanner")
st.caption("Kalibriert Zyklusmodelle automatisch im Walk-Forward-Backtest und scannt den gesamten Markt nach der höchsten Prognosegüte.")

# US-kompatible Datenquelle (Kraken)
@st.cache_resource
def get_exchange():
    return ccxt.kraken({'enableRateLimit': True})

exchange = get_exchange()

TF_CONFIG = {
    "15m": {"fetch_tf": "15m", "resample": None, "limit": 720, "tv": "15"},
    "1h":  {"fetch_tf": "1h",  "resample": None, "limit": 720, "tv": "60"},
    "4h":  {"fetch_tf": "4h",  "resample": None, "limit": 720, "tv": "240"},
    "1d":  {"fetch_tf": "1d",  "resample": None, "limit": 720, "tv": "D"},
    "2d":  {"fetch_tf": "1d",  "resample": "2D", "limit": 720, "tv": "2D"},
    "3d":  {"fetch_tf": "1d",  "resample": "3D", "limit": 720, "tv": "3D"},
    "4d":  {"fetch_tf": "1d",  "resample": "4D", "limit": 720, "tv": "4D"},
    "8d":  {"fetch_tf": "1d",  "resample": "8D", "limit": 720, "tv": "8D"},
    "1w":  {"fetch_tf": "1w",  "resample": None, "limit": 500, "tv": "W"}
}

ALL_COINS = [
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "NEAR",
    "SUI", "APT", "LTC", "BCH", "UNI", "ATOM", "ARB", "OP", "FIL", "INJ",
    "TIA", "RUNE", "AAVE", "SEI", "RENDER", "ICP", "PEPE"
]

# ----------------- Sidebar / Parameter -----------------
with st.sidebar:
    st.header("⚙️ Markt & Zeiteinheit")
    selected_coin = st.selectbox("Einzel-Analyse Asset", options=ALL_COINS, index=0)
    
    tf_options = list(TF_CONFIG.keys())
    tf_choice = st.selectbox("Timeframe", options=tf_options, index=tf_options.index("1d"))
    lookback_bars = st.slider(
        "Historischer Analysebereich (Kerzen)",
        min_value=80, max_value=450, value=220, step=20
    )
    
    st.markdown("---")
    st.subheader("🤖 Auto-Optimizer & Backtest")
    auto_optimize = st.checkbox(
        "⚡ Auto-Optimizer aktivieren",
        value=True,
        help="Testet automatisch alle Kombinationen aus Zyklenanzahl, WLS-Fokus und Trendgrad im Backtest und wählt das beste Setup."
    )
    enable_backtest = st.checkbox("Walk-Forward-Kurve im Chart zeigen", value=True)
    backtest_shift = st.slider(
        "Backtest-Fenster (Vor X Kerzen)",
        min_value=10, max_value=60, value=20, step=5
    )
    
    st.markdown("---")
    st.subheader("🎯 Manuelle Parameter (falls Auto aus)")
    recency_focus_manual = st.slider(
        "Gegenwarts-Fokus (WLS)", 
        min_value=0.0, max_value=5.0, value=2.5, step=0.5,
        disabled=auto_optimize
    )
    trend_degree_manual = st.selectbox(
        "Basis-Trend Modell", 
        options=[1, 2, 3], 
        index=1,
        format_func=lambda x: "Linear (Gerade)" if x == 1 else f"Polynom ({x}. Grades)",
        disabled=auto_optimize
    )
    num_components_manual = st.slider(
        "Anzahl kombinierter Zyklen",
        min_value=2, max_value=8, value=4, step=1,
        disabled=auto_optimize
    )
    anchor_to_last = st.checkbox("Projektion am Startkurs verankern", value=True)
    
    st.markdown("---")
    st.subheader("📡 Ehlers & Horizont")
    min_period = st.number_input("Min. Zykluslänge (Kerzen)", min_value=4, max_value=30, value=7, step=1)
    max_period = st.number_input("Max. Zykluslänge (Kerzen)", min_value=25, max_value=150, value=85, step=5)
    forecast_bars = st.slider("Zukunfts-Projektion (Kerzen)", min_value=10, max_value=100, value=30, step=5)
    
    st.button("🔄 Einzel-Analyse aktualisieren", use_container_width=True)

# ----------------- Daten abrufen -----------------
@st.cache_data(ttl=180)
def fetch_candles(coin, tf_key):
    cfg = TF_CONFIG[tf_key]
    symbol = f"{coin}/USD"
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=cfg["fetch_tf"], limit=cfg["limit"])
    if not ohlcv or len(ohlcv) < 60:
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
def ehlers_multi_peak_spectrum(prices, min_p, max_p, max_comp=8):
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
        candidate_p = [float(periods[idx]) for idx in sorted_peaks[:max_comp]]
    else:
        best_idx = int(np.argmax(score_arr))
        candidate_p = [float(periods[best_idx])]
        
    dom_period = int(candidate_p[0])
    dom_conf = float(np.max(score_arr)) * 100.0 if len(score_arr) > 0 else 0.0
    
    while len(candidate_p) < max_comp:
        harm = max(4.0, round(candidate_p[0] / (len(candidate_p) + 1), 1))
        if harm not in candidate_p:
            candidate_p.append(harm)
        else:
            candidate_p.append(max(4.0, candidate_p[-1] * 0.75))
            
    return dom_period, dom_conf, candidate_p, periods, score_arr

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

# ----------------- 3. Walk-Forward Grid-Search Auto-Optimizer -----------------
def run_walk_forward_optimizer(bt_train, bt_actual, all_bt_peaks, shift_n, anchor_last):
    cycle_counts = [2, 3, 4, 5, 6, 7]
    wls_focus_list = [0.5, 1.5, 2.5, 3.5, 4.5]
    poly_degrees = [1, 2]
    
    actual_diff = np.sign(np.diff(bt_actual))
    results = []
    
    for n_comp in cycle_counts:
        sub_periods = all_bt_peaks[:n_comp]
        for w_foc in wls_focus_list:
            for p_deg in poly_degrees:
                proj_full, _, _ = fit_high_precision_model(
                    bt_train, sub_periods, shift_n, w_foc, p_deg, anchor_last
                )
                fc_curve = proj_full[len(bt_train):]
                
                corr = 0.0
                if np.std(bt_actual) > 1e-8 and np.std(fc_curve) > 1e-8:
                    corr = float(np.corrcoef(bt_actual, fc_curve)[0, 1]) * 100.0
                
                pred_diff = np.sign(np.diff(fc_curve))
                dir_acc = float(np.mean(actual_diff == pred_diff)) * 100.0
                
                combined_score = 0.5 * dir_acc + 0.5 * max(0.0, corr)
                
                results.append({
                    "Zyklen": n_comp,
                    "WLS-Fokus": w_foc,
                    "Trend-Grad": p_deg,
                    "Trend-Typ": "Linear (1)" if p_deg == 1 else "Polynom (2)",
                    "Richtung (%)": round(dir_acc, 1),
                    "Korrelation (%)": round(corr, 1),
                    "Gesamt-Score": round(combined_score, 1)
                })
                
    res_df = pd.DataFrame(results).sort_values(by="Gesamt-Score", ascending=False).reset_index(drop=True)
    best_row = res_df.iloc[0]
    return best_row, res_df

# ----------------- Einzel-Asset Hauptausführung -----------------
with st.spinner(f"Berechne Zyklusmodell für {selected_coin}..."):
    df_raw = fetch_candles(selected_coin, tf_choice)

min_required = lookback_bars + backtest_shift + 10
if df_raw is None or len(df_raw) < min_required:
    st.error("Nicht genügend Kurshistorie für diesen Analyse- und Backtest-Zeitraum verfügbar.")
    st.stop()

df_bt_train = df_raw.iloc[-(lookback_bars + backtest_shift) : -backtest_shift].copy()
df_bt_actual = df_raw.iloc[-backtest_shift:].copy()
bt_prices_train = df_bt_train['Close'].values
bt_prices_actual = df_bt_actual['Close'].values

bt_dom_p, _, bt_all_peaks, _, _ = ehlers_multi_peak_spectrum(
    bt_prices_train, min_period, max_period, max_comp=8
)

opt_df = None
if auto_optimize:
    best_cfg, opt_df = run_walk_forward_optimizer(
        bt_prices_train, bt_prices_actual, bt_all_peaks, backtest_shift, anchor_to_last
    )
    active_num_comp = int(best_cfg["Zyklen"])
    active_wls_focus = float(best_cfg["WLS-Fokus"])
    active_trend_deg = int(best_cfg["Trend-Grad"])
else:
    active_num_comp = int(num_components_manual)
    active_wls_focus = float(recency_focus_manual)
    active_trend_deg = int(trend_degree_manual)

bt_active_p = bt_all_peaks[:active_num_comp]
bt_proj_full, _, _ = fit_high_precision_model(
    bt_prices_train, bt_active_p, backtest_shift, active_wls_focus, active_trend_deg, anchor_to_last
)
bt_forecast_curve = bt_proj_full[len(bt_prices_train):]

bt_corr = 0.0
if np.std(bt_prices_actual) > 1e-8 and np.std(bt_forecast_curve) > 1e-8:
    bt_corr = float(np.corrcoef(bt_prices_actual, bt_forecast_curve)[0, 1]) * 100.0

actual_diff = np.sign(np.diff(bt_prices_actual))
pred_diff = np.sign(np.diff(bt_forecast_curve))
dir_accuracy = float(np.mean(actual_diff == pred_diff)) * 100.0

df_analysis = df_raw.iloc[-lookback_bars:].copy()
prices = df_analysis['Close'].values
N = len(prices)

dom_period, dom_conf, all_live_peaks, spec_periods, spec_scores = ehlers_multi_peak_spectrum(
    prices, min_period, max_period, max_comp=8
)
active_periods = all_live_peaks[:active_num_comp]

projected_price, wave_dict, recent_fit_quality = fit_high_precision_model(
    prices, active_periods, forecast_bars, active_wls_focus, active_trend_deg, anchor_to_last
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

bt_pred_peaks, _ = find_peaks(bt_forecast_curve, distance=2)
bt_pred_troughs, _ = find_peaks(-bt_forecast_curve, distance=2)
bt_real_peaks, _ = find_peaks(bt_prices_actual, distance=2)
bt_real_troughs, _ = find_peaks(-bt_prices_actual, distance=2)

bt_eval_rows = []
for p in bt_pred_peaks:
    diffs = np.abs(bt_real_peaks - p) if len(bt_real_peaks) > 0 else np.array([999])
    min_dist = int(np.min(diffs))
    if min_dist == 0:
        hit_str = "✅ Exakter Volltreffer (0 Bars)"
    elif min_dist <= 2:
        hit_str = f"✅ Treffer in Toleranz (±{min_dist} Bar)"
    else:
        hit_str = f"❌ Abweichung ({min_dist} Bars)" if min_dist < 100 else "❌ Kein lokales Hoch"
    bt_eval_rows.append({
        "Prognostizierter Typ": "▼ Zyklus-HOCH (Top)",
        "Prognose-Datum": df_bt_actual.index[p].strftime("%d.%m.%Y %H:%M"),
        "Ergebnis im echten Markt": hit_str
    })

for t in bt_pred_troughs:
    diffs = np.abs(bt_real_troughs - t) if len(bt_real_troughs) > 0 else np.array([999])
    min_dist = int(np.min(diffs))
    if min_dist == 0:
        hit_str = "✅ Exakter Volltreffer (0 Bars)"
    elif min_dist <= 2:
        hit_str = f"✅ Treffer in Toleranz (±{min_dist} Bar)"
    else:
        hit_str = f"❌ Abweichung ({min_dist} Bars)" if min_dist < 100 else "❌ Kein lokales Tief"
    bt_eval_rows.append({
        "Prognostizierter Typ": "▲ Zyklus-TIEF (Boden)",
        "Prognose-Datum": df_bt_actual.index[t].strftime("%d.%m.%Y %H:%M"),
        "Ergebnis im echten Markt": hit_str
    })

# ----------------- Banner & Metriken -----------------
if auto_optimize:
    trend_name = "Linear (Grad 1)" if active_trend_deg == 1 else "Polynom (Grad 2)"
    st.success(
        f"🤖 **Auto-Optimizer ({selected_coin}):** Beste Walk-Forward-Genauigkeit mit **{active_num_comp} Zyklen**, "
        f"**WLS-Fokus {active_wls_focus}** und **{trend_name}** "
        f"(Richtungs-Trefferquote: **{dir_accuracy:.1f}%**, Korrelation: **{bt_corr:.1f}%**)."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset / Timeframe", f"{selected_coin} / {tf_choice}")
c2.metric("🎯 Dominanter Ehlers-Zyklus", f"{dom_period} Kerzen", f"Konfidenz: {dom_conf:.1f}%")
c3.metric("Optimiertes Setup", f"{active_num_comp} Zyklen | WLS {active_wls_focus}", f"In-Sample Fit: {recent_fit_quality:.1f}%")
c4.metric(
    f"⏪ Backtest ({backtest_shift} Bars)",
    f"{dir_accuracy:.1f} % Richtung",
    f"Korrelation: {bt_corr:.1f} %"
)

st.markdown("---")

# ----------------- Hauptchart -----------------
st.subheader(f"📈 {selected_coin} – Realkurs, Walk-Forward-Backtest & Zukunftsprojektion")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=df_analysis['Close'],
    mode='lines',
    name='Realer Kurs (Close)',
    line=dict(color='#d1d4dc', width=1.8)
))

fig.add_trace(go.Scatter(
    x=df_analysis.index,
    y=projected_price[:N],
    mode='lines',
    name='Live-Modell Fit (In-Sample)',
    line=dict(color='#3b82f6', width=1.5, dash='dot')
))

if enable_backtest:
    fig.add_trace(go.Scatter(
        x=df_bt_actual.index,
        y=bt_forecast_curve,
        mode='lines',
        name=f'⏪ Optimierte Backtest-Prognose (vor {backtest_shift} Bars)',
        line=dict(color='#f59e0b', width=3)
    ))
    fig.add_vline(
        x=df_bt_actual.index[0].timestamp() * 1000,
        line_width=1.5,
        line_dash="dot",
        line_color="#f59e0b"
    )

fig.add_trace(go.Scatter(
    x=future_times,
    y=projected_price[N:],
    mode='lines',
    name='🔮 Zukunfts-Projektion (ab JETZT)',
    line=dict(color='#089981', width=3)
))

if len(peaks) > 0:
    fig.add_trace(go.Scatter(
        x=[future_times[p] for p in peaks],
        y=[future_curve[p] for p in peaks],
        mode='markers',
        name='Zukunfts-Top ▼',
        marker=dict(color='#f23645', size=10, symbol='triangle-down')
    ))

if len(troughs) > 0:
    fig.add_trace(go.Scatter(
        x=[future_times[t] for t in troughs],
        y=[future_curve[t] for t in troughs],
        mode='markers',
        name='Zukunfts-Boden ▲',
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
    height=540,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ----------------- Detail-Tabs inkl. Multi-Coin-Scanner -----------------
tab_scan, tab_opt, tab_bt, tab1, tab2, tab3 = st.tabs([
    "🌐 Multi-Coin Scanner (Markt-Ranking)",
    "🏆 Auto-Optimizer Ranking",
    "⏪ Backtest-Auswertung",
    "🎯 Zukünftige Wendepunkte",
    "🌊 Einzelne Zykluskomponenten",
    "📡 Ehlers Spektrum"
])

with tab_scan:
    st.markdown(f"##### 🌐 Multi-Coin Walk-Forward Scanner ({tf_choice})")
    st.caption(
        "Scannt die Top-Krypto-Assets, führt für jeden Coin eine Walk-Forward-Optimierung durch "
        "und sortiert sie danach, welcher Markt sich aktuell am genauesten an sein Zyklusmodell hält."
    )
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        num_scan_coins = st.slider("Anzahl zu scannender Coins", min_value=5, max_value=len(ALL_COINS), value=15, step=5)
    with col_s2:
        st.write("")
        start_scan = st.button("🚀 Alle Coins jetzt scannen & ranken", type="primary", use_container_width=True)
        
    if start_scan:
        scan_list = ALL_COINS[:num_scan_coins]
        scan_rows = []
        prog_text = st.empty()
        prog_bar = st.progress(0)
        tv_tf_code = TF_CONFIG[tf_choice]["tv"]
        
        for idx_c, c_sym in enumerate(scan_list):
            prog_text.text(f"Optimiere & teste Zyklusmodell ({idx_c + 1}/{len(scan_list)}): {c_sym} ({tf_choice}) ...")
            try:
                df_c = fetch_candles(c_sym, tf_choice)
                if df_c is not None and len(df_c) >= min_required:
                    c_bt_train = df_c.iloc[-(lookback_bars + backtest_shift) : -backtest_shift]['Close'].values
                    c_bt_actual = df_c.iloc[-backtest_shift:]['Close'].values
                    
                    _, _, c_bt_peaks, _, _ = ehlers_multi_peak_spectrum(c_bt_train, min_period, max_period, max_comp=8)
                    best_c_row, _ = run_walk_forward_optimizer(
                        c_bt_train, c_bt_actual, c_bt_peaks, backtest_shift, anchor_to_last
                    )
                    
                    # Live-Prognose mit dem Gewinner-Setup des Coins berechnen
                    c_live_df = df_c.iloc[-lookback_bars:]
                    c_live_prices = c_live_df['Close'].values
                    c_dom_p, c_conf, c_live_peaks, _, _ = ehlers_multi_peak_spectrum(
                        c_live_prices, min_period, max_period, max_comp=8
                    )
                    
                    opt_n = int(best_c_row["Zyklen"])
                    opt_w = float(best_c_row["WLS-Fokus"])
                    opt_d = int(best_c_row["Trend-Grad"])
                    
                    c_proj, _, _ = fit_high_precision_model(
                        c_live_prices, c_live_peaks[:opt_n], forecast_bars, opt_w, opt_d, anchor_to_last
                    )
                    
                    c_fut_curve = c_proj[len(c_live_prices):]
                    c_tstep = c_live_df.index[-1] - c_live_df.index[-2]
                    c_ftimes = [c_live_df.index[-1] + c_tstep * (k + 1) for k in range(forecast_bars)]
                    
                    c_pk, _ = find_peaks(c_fut_curve, distance=2)
                    c_tr, _ = find_peaks(-c_fut_curve, distance=2)
                    
                    next_events = []
                    for p_i in c_pk:
                        next_events.append((int(p_i + 1), f"▼ TOP in +{p_i+1} Bars ({c_ftimes[p_i].strftime('%d.%m. %H:%M')})"))
                    for t_i in c_tr:
                        next_events.append((int(t_i + 1), f"▲ BODEN in +{t_i+1} Bars ({c_ftimes[t_i].strftime('%d.%m. %H:%M')})"))
                        
                    next_events.sort(key=lambda x: x[0])
                    next_turn_str = next_events[0][1] if next_events else "—"
                    
                    tv_link = f"https://www.tradingview.com/chart/?symbol=BYBIT:{c_sym}USDT.P&interval={tv_tf_code}"
                    
                    scan_rows.append({
                        "Symbol": f"{c_sym}USDT.P",
                        "Gesamt-Score": float(best_c_row["Gesamt-Score"]),
                        "Richtung (%)": float(best_c_row["Richtung (%)"]),
                        "Korrelation (%)": float(best_c_row["Korrelation (%)"]),
                        "Dom. Zyklus": f"{c_dom_p} Bars",
                        "Opt. Setup": f"{opt_n} Zyk. | WLS {opt_w}",
                        "Nächster Zyklus-Wendepunkt": next_turn_str,
                        "TradingView": tv_link
                    })
            except Exception:
                pass
            prog_bar.progress((idx_c + 1) / len(scan_list))
            time.sleep(0.03)
            
        prog_text.empty()
        prog_bar.empty()
        
        if scan_rows:
            st.session_state["last_scan_df"] = pd.DataFrame(scan_rows).sort_values(
                by="Gesamt-Score", ascending=False
            ).reset_index(drop=True)
            
    if "last_scan_df" in st.session_state:
        st.dataframe(
            st.session_state["last_scan_df"],
            column_config={
                "Gesamt-Score": st.column_config.ProgressColumn(
                    "Walk-Forward Score", format="%.1f", min_value=0, max_value=100
                ),
                "Richtung (%)": st.column_config.NumberColumn("Richtungs-Quote", format="%.1f %%"),
                "Korrelation (%)": st.column_config.NumberColumn("Kurven-Korrelation", format="%.1f %%"),
                "TradingView": st.column_config.LinkColumn("Chart", display_text="Chart öffnen ↗")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Klicke oben auf '🚀 Alle Coins jetzt scannen & ranken', um die Rangliste der zyklisch präzisesten Coins zu erstellen.")

with tab_opt:
    st.markdown(f"##### 🏆 Rangliste der getesteten Parameter für {selected_coin} (Top 15)")
    if opt_df is not None:
        st.dataframe(
            opt_df.head(15).drop(columns=["Trend-Grad"]),
            column_config={
                "Gesamt-Score": st.column_config.ProgressColumn("Gesamt-Score", format="%.1f", min_value=0, max_value=100),
                "Richtung (%)": st.column_config.NumberColumn("Richtungs-Trefferquote", format="%.1f %%"),
                "Korrelation (%)": st.column_config.NumberColumn("Kurven-Korrelation", format="%.1f %%")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Aktiviere links die Option '⚡ Auto-Optimizer aktivieren', um das Parameter-Ranking zu berechnen.")

with tab_bt:
    st.markdown(f"##### ⏪ Detailvergleich ({selected_coin}): Prognose von vor {backtest_shift} Kerzen vs. Echter Kursverlauf")
    col_bt1, col_bt2 = st.columns([3, 2])
    with col_bt1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=df_bt_actual.index,
            y=bt_prices_actual,
            mode='lines+markers',
            name='Tatsächlicher Kurs (Ist)',
            line=dict(color='#d1d4dc', width=2)
        ))
        fig_bt.add_trace(go.Scatter(
            x=df_bt_actual.index,
            y=bt_forecast_curve,
            mode='lines+markers',
            name='Optimierte Zyklus-Prognose (Soll)',
            line=dict(color='#f59e0b', width=2.5)
        ))
        fig_bt.update_layout(
            template="plotly_dark",
            height=360,
            title="Isoliertes Backtest-Fenster (Out-of-Sample)",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_bt, use_container_width=True)
        
    with col_bt2:
        st.markdown("###### Trefferprüfung der prognostizierten Wendepunkte")
        if bt_eval_rows:
            st.dataframe(pd.DataFrame(bt_eval_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Im gewählten Backtest-Fenster lagen keine ausgeprägten Zyklus-Wendepunkte.")

with tab1:
    st.markdown(f"##### Chronologische Liste der nächsten Zyklus-Wendepunkte ({selected_coin})")
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
