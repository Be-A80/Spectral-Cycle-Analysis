import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, date, time as dt_time

# ----------------- Seiten-Konfiguration -----------------
st.set_page_config(page_title="PZP Cycle Profiler (Ehlers)", page_icon="🎯", layout="wide")

st.title("🎯 PZP Zyklus-Profiler mit Ehlers Dominant Cycle Finder")
st.caption("Kombiniert die objektive Erkennung dominanter Zyklen nach John Ehlers mit der statistischen PZP-Phasenmatrix und ±1-Toleranz.")

# US-kompatible Datenquelle (Kraken)
@st.cache_resource
def get_exchange():
    return ccxt.kraken({
        'enableRateLimit': True,
    })

exchange = get_exchange()

# Timeframe-Konfigurationen
TF_CONFIG = {
    "15m": {"fetch_tf": "15m", "resample": None,  "limit": 720},
    "30m": {"fetch_tf": "30m", "resample": None,  "limit": 720},
    "1h":  {"fetch_tf": "1h",  "resample": None,  "limit": 720},
    "2h":  {"fetch_tf": "1h",  "resample": "2h",  "limit": 720},
    "4h":  {"fetch_tf": "4h",  "resample": None,  "limit": 720},
    "1d":  {"fetch_tf": "1d",  "resample": None,  "limit": 720},
    "2d":  {"fetch_tf": "1d",  "resample": "2D",  "limit": 720},
    "3d":  {"fetch_tf": "1d",  "resample": "3D",  "limit": 720},
    "4d":  {"fetch_tf": "1d",  "resample": "4D",  "limit": 720},
    "1w":  {"fetch_tf": "1w",  "resample": None,  "limit": 500}
}

# ----------------- Ehlers Autokorrelations-Algorithmus -----------------
def find_dominant_cycle_ehlers(prices, min_p=6, max_p=80):
    """
    Ermittelt den dominanten Zyklus mittels Ehlers-Autokorrelation.
    """
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
    return int(periods[best_idx]), float(corr_arr[best_idx]) * 100.0, periods, corr_arr

# ----------------- Daten abrufen -----------------
def fetch_candles(coin, tf_choice):
    cfg = TF_CONFIG[tf_choice]
    symbol = f"{coin}/USD"
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=cfg["fetch_tf"], limit=cfg["limit"])
    if not ohlcv or len(ohlcv) < 30:
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

# ----------------- PZP Trendwechsel-Erkennung -----------------
def identify_trend_changes(df, L, R, f):
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    has_turn = np.zeros(n, dtype=bool)
    ph_indices = []
    pl_indices = []
    
    for i in range(L, n - R):
        if all(highs[i] > highs[i - k] for k in range(1, L + 1)) and all(highs[i] > highs[i + k] for k in range(1, R + 1)):
            ph_indices.append(i)
            if len(ph_indices) >= 2:
                dist = np.ceil(f * (ph_indices[-1] - ph_indices[-2]))
                target_bar = int(ph_indices[-1] + dist + R)
                if target_bar < n:
                    has_turn[target_bar] = True
                    
        if all(lows[i] < lows[i - k] for k in range(1, L + 1)) and all(lows[i] < lows[i + k] for k in range(1, R + 1)):
            pl_indices.append(i)
            if len(pl_indices) >= 2:
                dist = np.ceil(f * (pl_indices[-1] - pl_indices[-2]))
                target_bar = int(pl_indices[-1] + dist + R)
                if target_bar < n:
                    has_turn[target_bar] = True

    df['has_turn'] = has_turn
    return df

# ----------------- Sidebar / Parameter -----------------
with st.sidebar:
    st.header("⚙️ Markt & Asset")
    market_list = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "NEAR"]
    selected_coin = st.selectbox("Markt / Asset", options=market_list, index=0)
    
    tf_options = list(TF_CONFIG.keys())
    tf_choice = st.selectbox("Timeframe", options=tf_options, index=tf_options.index("1d"))
    
    st.markdown("---")
    st.subheader("📡 Ehlers Cycle-Steuerung")
    use_ehlers_auto = st.checkbox("🎯 Ehlers Dominant Cycle automatisch nutzen", value=True, help="Berechnet per Autokorrelation die stärkste Periodenlänge und setzt diese als Zykluslänge ein.")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        scan_min = st.number_input("Scan Min", min_value=4, max_value=20, value=6, step=1)
    with col_p2:
        scan_max = st.number_input("Scan Max", min_value=20, max_value=120, value=60, step=5)
        
    manual_cycle_len = st.number_input("Manuelle Zykluslänge N (falls Auto deaktiviert)", min_value=3, max_value=120, value=18, step=1)
    cycle_count = st.number_input("Anzahl Zyklen zum Vergleichen", min_value=2, max_value=40, value=10, step=1)
    
    st.markdown("##### ⏱️ Toleranz & Start")
    use_tolerance = st.checkbox("±1-Kerze Toleranz einbeziehen", value=True)
    
    col_d, col_t = st.columns([1, 1])
    with col_d:
        start_date_input = st.date_input("Startdatum", value=date(2024, 1, 1))
    with col_t:
        start_time_input = st.time_input("Startuhrzeit (UTC)", value=dt_time(0, 0), step=300)
        
    auto_align = st.checkbox("🎯 Auto-Phase-Alignment (Startpunkt optimieren)", value=True)
    manual_offset = 0
    if not auto_align:
        manual_offset = st.slider("Manueller Start-Offset (Kerzen)", 0, 30, 0)
        
    st.markdown("---")
    st.subheader("📐 PZP-Parameter")
    pL = st.number_input("Pivot Links (L)", min_value=1, value=2)
    pR = st.number_input("Pivot Rechts (R)", min_value=1, value=2)
    factor = st.number_input("Projektionsfaktor", min_value=0.05, value=0.5, step=0.05)
    
    st.button("🔄 Analyse ausführen", use_container_width=True)

# ----------------- Hauptausführung -----------------
with st.spinner(f"Lade Kursdaten für {selected_coin}..."):
    df_raw = fetch_candles(selected_coin, tf_choice)

if df_raw is None or len(df_raw) < 50:
    st.error("Fehler beim Laden der Kursdaten.")
    st.stop()

# 1. Ehlers Dominanten Zyklus berechnen
dom_period, confidence, all_periods, all_corrs = find_dominant_cycle_ehlers(df_raw['Close'].values, scan_min, scan_max)

# Festlegen, welche Zykluslänge verwendet wird
if use_ehlers_auto:
    cycle_length = dom_period
else:
    cycle_length = int(manual_cycle_len)

# 2. Trendwechsel mit PZP identifizieren
df = identify_trend_changes(df_raw, pL, pR, factor)

# Startdatum anwenden
start_dt = pd.to_datetime(datetime.combine(start_date_input, start_time_input)).tz_localize('UTC')
df_sub = df[df.index >= start_dt].copy()

if len(df_sub) < (cycle_count * cycle_length + cycle_length):
    available_cycles = (len(df_sub) - cycle_length) // cycle_length
    if available_cycles < 2:
        st.error("Zu wenige Kerzen ab dem gewählten Startzeitpunkt vorhanden. Bitte wähle ein früheres Startdatum.")
        st.stop()
    cycle_count = available_cycles

# 3. Phasen-Optimierung (Offset finden)
phase_eval = []

for offset in range(int(cycle_length)):
    sub_slice = df_sub.iloc[offset : offset + cycle_count * cycle_length].copy()
    sub_slice['Position'] = [idx % cycle_length for idx in range(len(sub_slice))]
    sub_slice['Cycle_ID'] = [idx // cycle_length + 1 for idx in range(len(sub_slice))]
    
    max_rate = 0.0
    best_pos = 0
    total_score = 0.0
    
    for pos in range(cycle_length):
        if use_tolerance:
            valid_pos = [(pos - 1) % cycle_length, pos, (pos + 1) % cycle_length]
            turns = sub_slice[(sub_slice['Position'].isin(valid_pos)) & (sub_slice['has_turn'] == True)]
        else:
            turns = sub_slice[(sub_slice['Position'] == pos) & (sub_slice['has_turn'] == True)]
            
        hit_cycles = turns['Cycle_ID'].nunique()
        pct = (hit_cycles / cycle_count) * 100
        total_score += pct**2
        if pct > max_rate:
            max_rate = pct
            best_pos = pos
            
    start_time_actual = sub_slice.index[0]
    phase_eval.append({
        "Offset (Kerzen)": offset,
        "Effektiver Start": start_time_actual.strftime("%d.%m.%y %H:%M"),
        "Spitzen-Trefferquote (%)": round(max_rate, 1),
        "Beste Position": best_pos,
        "Score": total_score
    })

phase_df = pd.DataFrame(phase_eval)

if auto_align:
    best_phase_row = phase_df.sort_values(by=["Spitzen-Trefferquote (%)", "Score"], ascending=False).iloc[0]
    chosen_offset = int(best_phase_row["Offset (Kerzen)"])
else:
    chosen_offset = manual_offset % cycle_length
    best_phase_row = phase_df[phase_df["Offset (Kerzen)"] == chosen_offset].iloc[0]

# 4. Finale Analyse mit gewähltem Offset
df_final = df_sub.iloc[chosen_offset : chosen_offset + cycle_count * cycle_length].copy()
df_final['Position'] = [idx % cycle_length for idx in range(len(df_final))]
df_final['Cycle_ID'] = [idx // cycle_length + 1 for idx in range(len(df_final))]

position_stats = []
for pos in range(cycle_length):
    exact_turns = df_final[(df_final['Position'] == pos) & (df_final['has_turn'] == True)]
    exact_cycles = exact_turns['Cycle_ID'].nunique()
    exact_pct = round((exact_cycles / cycle_count) * 100, 1)
    
    valid_pos = [(pos - 1) % cycle_length, pos, (pos + 1) % cycle_length]
    tol_turns = df_final[(df_final['Position'].isin(valid_pos)) & (df_final['has_turn'] == True)]
    tol_cycles = tol_turns['Cycle_ID'].nunique()
    tol_pct = round((tol_cycles / cycle_count) * 100, 1)
    
    position_stats.append({
        "Position": pos,
        "Exakt (%)": exact_pct,
        "Mit ±1 Toleranz (%)": tol_pct,
        "Treffer Zyklen (Toleranz)": f"{tol_cycles} / {cycle_count}"
    })

stats_df = pd.DataFrame(position_stats)
display_metric = "Mit ±1 Toleranz (%)" if use_tolerance else "Exakt (%)"

# ----------------- UI Dashboard -----------------
st.info(f"📡 **Ehlers Autokorrelation:** Dominanter Markt-Zyklus bei **{dom_period} Kerzen** (Konfidenz: **{confidence:.1f}%**). Aktive Zykluslänge: **{cycle_length} Kerzen**.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Zykluslänge", f"{cycle_length} Kerzen ({tf_choice})")
col2.metric("Gewählter Offset", f"+{chosen_offset} Kerzen")
col3.metric("Effektiver Start", f"{df_final.index[0].strftime('%d.%m.%y %H:%M')}")
top_res = stats_df.sort_values(by=display_metric, ascending=False).iloc[0]
col4.metric(f"Top-Trefferpunkt", f"Pos {int(top_res['Position'])}", f"{top_res[display_metric]}% Treffer")

st.markdown("---")

# Vertrautes Balkendiagramm: Exakt vs. mit ±1 Toleranz
st.subheader("📊 Wahrscheinlichkeitsverteilung über den Zyklus")
plot_df = stats_df.melt(id_vars=["Position"], value_vars=["Exakt (%)", "Mit ±1 Toleranz (%)"], 
                        var_name="Methode", value_name="Quote")

fig = px.bar(
    plot_df,
    x="Position",
    y="Quote",
    color="Methode",
    barmode="group",
    text="Quote",
    labels={"Position": f"Kerzenposition im Zyklus (0 bis {cycle_length - 1})", "Quote": "Wahrscheinlichkeit (%)"},
    title=f"Zyklus-Verteilung ({selected_coin} / {tf_choice} – {cycle_count} Zyklen à {cycle_length} Kerzen)",
    color_discrete_map={"Exakt (%)": "#3b82f6", "Mit ±1 Toleranz (%)": "#089981"}
)
fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
fig.update_layout(
    yaxis=dict(range=[0, 115]),
    xaxis=dict(tickmode='linear', dtick=1),
    template="plotly_dark",
    height=450
)
st.plotly_chart(fig, use_container_width=True)

# Tabs: Rangliste & Matrix, Phasenvergleich, Ehlers Spektrum
tab1, tab2, tab3 = st.tabs(["🏆 Rangliste & Matrix", "🔬 Phasen-Vergleich", "📡 Ehlers Periodogramm"])

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Rangliste der Positionen")
        st.dataframe(
            stats_df.sort_values(by=display_metric, ascending=False),
            column_config={
                "Position": st.column_config.NumberColumn("Position", format="%d"),
                "Exakt (%)": st.column_config.ProgressColumn("Exakt", format="%.1f%%", min_value=0, max_value=100),
                "Mit ±1 Toleranz (%)": st.column_config.ProgressColumn("Mit Toleranz (±1)", format="%.1f%%", min_value=0, max_value=100)
            },
            use_container_width=True,
            hide_index=True
        )
    with c2:
        st.markdown("##### Treffer-Matrix je Zyklus (Exakte Lage)")
        piv = pd.crosstab(
            index=df_final['Cycle_ID'],
            columns=df_final['Position'],
            values=df_final['has_turn'],
            aggfunc=lambda x: "✅" if any(x) else "—"
        ).fillna("—")
        piv.index.name = "Zyklus #"
        st.dataframe(piv, use_container_width=True)

with tab2:
    st.markdown("##### Vergleich aller getesteten Start-Offsets")
    st.caption("Zeigt die maximale Trefferquote für jede Startpunktverschiebung:")
    st.dataframe(
        phase_df.drop(columns=["Score"]),
        column_config={
            "Spitzen-Trefferquote (%)": st.column_config.ProgressColumn("Spitzenquote", format="%.1f%%", min_value=0, max_value=100),
            "Offset (Kerzen)": st.column_config.NumberColumn("Verschiebung", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.markdown("##### Autokorrelations-Spektrum nach John Ehlers")
    st.caption(f"Der grüne Balken zeigt den erkannten dominanten Zyklus ({dom_period} Kerzen):")
    spec_df = pd.DataFrame({"Periode": all_periods, "Korrelation": all_corrs})
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
