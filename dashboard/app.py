# =============================================================================
#  Run: streamlit run dashboard/app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt 
import sys 
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai.main import run_pipeline
from ai.gnn import draw_attention_graph



# =============================================================================
#  PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Adaptive Lab Guardian",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  PATHS
# =============================================================================
LOG_FILE     = os.path.join(os.path.dirname(__file__), "..", "data", "sensor_log.csv")
SENSOR_TOPIC = "alg1/sensors"
ACTION_TOPIC = "alg1/actions"

# =============================================================================
#  CUSTOM CSS — matches the dark industrial design
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg:       #0c0f18;
    --bg2:      #111827;
    --bg3:      #1a2236;
    --border:   #1e2d45;
    --border2:  #243352;
    --green:    #00e5a0;
    --green-dim:#00b37d;
    --green-bg: rgba(0,229,160,0.10);
    --yellow:   #f5b731;
    --yellow-dim:#c8941f;
    --yellow-bg:rgba(245,183,49,0.10);
    --red:      #ff4455;
    --red-dim:  #cc2233;
    --red-bg:   rgba(255,68,85,0.10);
    --blue:     #4d9fff;
    --blue-bg:  rgba(77,159,255,0.10);
    --text:     #e8edf8;
    --text2:    #8899bb;
    --text3:    #3d4f6e;
    --mono:     'JetBrains Mono', monospace;
    --display:  'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--mono);
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }
[data-testid="stSidebarContent"] { padding: 1.2rem 0.8rem !important; }

/* ── Sidebar radio ── */
.stRadio > label { display: none !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text2) !important; font-size: 0.72rem !important; }
.stRadio label { color: var(--text2) !important; font-size: 0.72rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--bg3) !important;
    color: var(--blue) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 0.4rem 1rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--blue-bg) !important;
    border-color: var(--blue) !important;
}

/* ── Checkbox ── */
.stCheckbox label { color: var(--text2) !important; font-size: 0.72rem !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* ── Streamlit metric override ── */
[data-testid="stMetric"] { display: none; }

/* ── Custom cards ── */
.alg-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.8rem;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 0.6rem;
}
.alg-title {
    font-family: var(--display);
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.alg-sub { font-size: 0.65rem; color: var(--text2); margin-top: 2px; }
.alg-topbar-right { display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap; }
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--green-bg);
    border: 1px solid var(--green-dim);
    border-radius: 20px;
    padding: 0.28rem 0.8rem;
    font-size: 0.65rem;
    color: var(--green);
    font-weight: 700;
    letter-spacing: 0.08em;
}
.live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s ease-in-out infinite;
    display: inline-block;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
.time-pill {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-size: 0.65rem;
    color: var(--text2);
}
.state-pill-normal  { color: var(--green); border-color: var(--green-dim); background: var(--green-bg); }
.state-pill-warning { color: var(--yellow); border-color: var(--yellow-dim); background: var(--yellow-bg); }
.state-pill-danger  { color: var(--red); border-color: var(--red-dim); background: var(--red-bg); }

/* section header */
.sec-hdr {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    color: var(--text3);
    text-transform: uppercase;
    font-weight: 700;
    padding: 1.1rem 1.8rem 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.8rem;
}
.sec-hdr::before { content: '◆'; color: var(--blue); font-size: 0.5rem; }

/* sensor card */
.s-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 0.8rem;
    text-align: center;
    transition: border-color 0.2s;
    height: 100%;
}
.s-card:hover { border-color: var(--border2); }
.s-icon-wrap {
    width: 38px; height: 38px;
    border-radius: 10px;
    margin: 0 auto 0.5rem;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.s-label { font-size: 0.62rem; color: var(--text2); letter-spacing: 0.05em; margin-bottom: 0.3rem; }
.s-value { font-family: var(--display); font-size: 1.5rem; font-weight: 800; color: var(--text); }
.s-unit  { font-size: 0.6rem; color: var(--text3); margin-top: 1px; }

/* pipeline card */
.p-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    height: 100%;
}
.p-card-accent-green { border-left: 3px solid var(--green); }
.p-card-accent-blue  { border-left: 3px solid var(--blue); }
.p-card-accent-yellow{ border-left: 3px solid var(--yellow); }
.p-card-accent-red   { border-left: 3px solid var(--red); }
.p-lbl { font-size: 0.58rem; letter-spacing: 0.1em; color: var(--text2); text-transform: uppercase; font-weight: 700; margin-bottom: 0.4rem; }
.p-val-green  { font-family: var(--display); font-size: 1.25rem; font-weight: 800; color: var(--green); }
.p-val-yellow { font-family: var(--display); font-size: 1.25rem; font-weight: 800; color: var(--yellow); }
.p-val-red    { font-family: var(--display); font-size: 1.25rem; font-weight: 800; color: var(--red); }
.p-val-blue   { font-family: var(--display); font-size: 1.25rem; font-weight: 800; color: var(--blue); }
.p-stat-num   { font-family: var(--display); font-size: 2rem; font-weight: 800; line-height: 1; }

/* badge */
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.b-green  { background: var(--green-bg);  color: var(--green);  border: 1px solid var(--green-dim); }
.b-yellow { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow-dim); }
.b-red    { background: var(--red-bg);    color: var(--red);    border: 1px solid var(--red-dim); }
.b-blue   { background: var(--blue-bg);   color: var(--blue);   border: 1px solid var(--blue); }

/* actuator card */
.act-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
    height: 100%;
}
.act-icon  { font-size: 20px; margin-bottom: 0.25rem; }
.act-label { font-size: 0.58rem; color: var(--text2); letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700; margin-bottom: 0.3rem; }
.act-state { font-family: var(--display); font-size: 0.85rem; font-weight: 800; }

/* chart card */
.ch-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem 0.5rem;
    height: 100%;
}
.ch-title { font-size: 0.65rem; color: var(--text2); letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.5rem; }

/* event log */
.ev-table {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}
.ev-hdr {
    display: grid;
    grid-template-columns: 85px 1fr 100px 180px;
    gap: 8px;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.6rem;
    color: var(--text3);
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.ev-row {
    display: grid;
    grid-template-columns: 85px 1fr 100px 180px;
    gap: 8px;
    padding: 0.55rem 1rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.68rem;
    color: var(--text2);
    align-items: center;
}
.ev-row:last-child { border-bottom: none; }
.ev-time { color: var(--blue); font-size: 0.65rem; }

/* Sidebar styles */
.sb-logo-title { font-family: var(--display); font-size: 0.9rem; font-weight: 800; color: var(--text); }
.sb-logo-sub   { font-size: 0.58rem; color: var(--text3); line-height: 1.5; margin-top: 2px; }
.sb-section    { font-size: 0.58rem; letter-spacing: 0.14em; color: var(--text3); text-transform: uppercase; font-weight: 700; margin: 0.8rem 0 0.3rem; }
.sb-info-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem;
    font-size: 0.62rem;
    color: var(--text2);
    line-height: 1.8;
    margin-top: 1rem;
}
.sb-key { color: var(--green); font-weight: 700; font-size: 0.6rem; }

.content-pad { padding: 0 1.8rem 1.8rem; }

/* Streamlit area_chart adjustments */
[data-testid="stVegaLiteChart"] {
    border-radius: 6px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
#  SIMULATION HELPERS
# =============================================================================

def _simulate_sensor() -> dict:
    scenario = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
    if scenario == 0:
        return {"Temp_C": round(random.uniform(20, 27), 2),
                "Humidity_pct": round(random.uniform(40, 60), 1),
                "Gas_AQI": round(random.uniform(80, 200), 1),
                "Light_Lux": round(random.uniform(300, 700), 1),
                "Motion_Detected": random.choice([0, 0, 1])}
    elif scenario == 1:
        return {"Temp_C": round(random.uniform(28, 36), 2),
                "Humidity_pct": round(random.uniform(60, 75), 1),
                "Gas_AQI": round(random.uniform(300, 550), 1),
                "Light_Lux": round(random.uniform(100, 300), 1),
                "Motion_Detected": random.choice([0, 1])}
    else:
        return {"Temp_C": round(random.uniform(37, 48), 2),
                "Humidity_pct": round(random.uniform(75, 95), 1),
                "Gas_AQI": round(random.uniform(600, 950), 1),
                "Light_Lux": round(random.uniform(0, 80), 1),
                "Motion_Detected": 1}


def _simulate_meta(sensor: dict) -> dict:
    gas_norm = sensor["Gas_AQI"] / 1000.0
    if gas_norm > 0.6 or sensor["Temp_C"] > 36:
        state, label = 2, "Dangerous"
    elif gas_norm > 0.3 or sensor["Temp_C"] > 27:
        state, label = 1, "Warning"
    else:
        state, label = 0, "Normal"
    actions = {"fan": "OFF", "alarm": "OFF", "servo": "CLOSED", "buzzer": "OFF", "rgb_led": "GREEN"}
    if state == 2:
        actions = {"fan": "ON", "alarm": "ON", "servo": "OPEN", "buzzer": "ON", "rgb_led": "RED"}
    elif state == 1:
        actions = {"fan": "ON", "alarm": "OFF", "servo": "OPEN", "buzzer": "OFF", "rgb_led": "YELLOW"}
    return {
        "_meta": {"state": state, "state_label": label,
                  "gas_pred": round(gas_norm + random.uniform(-0.03, 0.03), 4),
                  "temp_pred": round(sensor["Temp_C"] / 50 + random.uniform(-0.02, 0.02), 4),
                  "is_anomaly": random.random() < 0.05,
                  "reward": round(random.uniform(-0.5, 1.0), 2)},
        **actions,
    }


def _make_sim_history(n: int = 40) -> pd.DataFrame:
    rows = []
    base = datetime.now() - timedelta(minutes=n)
    for i in range(n):
        s = _simulate_sensor()
        m = _simulate_meta(s)
        rows.append({
            "timestamp": (base + timedelta(minutes=i)).strftime("%H:%M:%S"),
            **s,
            "state_label": m["_meta"]["state_label"],
            "fan": m["fan"], "alarm": m["alarm"],
            "servo": m["servo"], "buzzer": m["buzzer"],
            "rgb_led": m["rgb_led"],
            "is_anomaly": m["_meta"]["is_anomaly"],
            "gas_pred": m["_meta"]["gas_pred"],
            "temp_pred": m["_meta"]["temp_pred"],
            "reward": m["_meta"]["reward"],
        })
    return pd.DataFrame(rows)


def load_data(mode: str) -> pd.DataFrame:
    if mode == "csv":
        try:
            if not os.path.exists(LOG_FILE):
                st.sidebar.warning("⚠ sensor_log.csv not found — using demo data.")
                return _make_sim_history()
            df = pd.read_csv(LOG_FILE)
            return df.tail(40).reset_index(drop=True) if not df.empty else _make_sim_history()
        except Exception as e:
            st.sidebar.error(f"CSV read error: {e}")
            return _make_sim_history()
    return _make_sim_history()


def badge(label: str) -> str:
    mapping = {
        "normal":    ("b-green",  label),
        "warning":   ("b-yellow", label),
        "dangerous": ("b-red",    label),
        "true":      ("b-red",    "⚠ ANOMALY"),
        "false":     ("b-green",  "✔ Known"),
        "on":        ("b-red",    "ON"),
        "off":       ("b-blue",   "OFF"),
        "open":      ("b-green",  "OPEN"),
        "closed":    ("b-blue",   "CLOSED"),
        "red":       ("b-red",    "🔴 RED"),
        "yellow":    ("b-yellow", "🟡 YELLOW"),
        "green":     ("b-green",  "🟢 GREEN"),
    }
    cls, txt = mapping.get(str(label).lower(), ("b-blue", label))
    return f'<span class="badge {cls}">{txt}</span>'


# =============================================================================
#  SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0.3rem 1rem;border-bottom:1px solid var(--border);margin-bottom:0.8rem;">
        <div style="width:36px;height:36px;background:rgba(0,229,160,0.1);border:1px solid #00b37d;
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;margin-bottom:0.5rem;">🧪</div>
        <div class="sb-logo-title">Adaptive Lab Guardian</div>
        <div class="sb-logo-sub">Smart Adaptive Environment<br>Monitoring &amp; Decision System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    nav_items = [("⊞", "Dashboard"), ("◎", "Sensors"), ("⬡", "AI Pipeline"),
                 ("◈", "Actions"), ("📈", "Charts"), ("≡", "Event Log"),
                 ("⚙", "Settings"), ("ℹ", "About")]
    for icon, name in nav_items:
        active = "background:rgba(77,159,255,0.15);color:var(--blue);border:1px solid rgba(77,159,255,0.25);" if name == "Dashboard" else ""
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.42rem 0.6rem;'
            f'border-radius:6px;cursor:pointer;color:var(--text2);font-size:0.7rem;'
            f'font-weight:600;letter-spacing:0.04em;{active}">'
            f'<span style="width:16px;text-align:center;">{icon}</span>{name}</div>',
            unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Data Source</div>', unsafe_allow_html=True)
    data_mode = st.radio("src", ["Demo Mode", "Read from sensor_log.csv"], label_visibility="collapsed")
    mode_key = "demo" if data_mode == "Demo Mode" else "csv"

    st.markdown("---")
    if st.button("↺  Refresh Now"):
        st.rerun()
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)

    st.markdown(f"""
    <div class="sb-info-box">
        <span class="sb-key">BROKER</span><br>
        broker.hivemq.com:1883<br><br>
        <span class="sb-key">TOPICS</span><br>
        SUB → alg1/sensors<br>
        PUB → alg1/actions<br><br>
        <span class="sb-key">MODE</span><br>
        {data_mode}
    </div>""", unsafe_allow_html=True)

# =============================================================================
#  LOAD DATA
# =============================================================================
df  = load_data(mode_key)
rec = df.iloc[-1].to_dict()
now = datetime.now().strftime("%b %d, %Y · %I:%M %p")

temp      = rec.get("Temp_C", "--")
humidity  = rec.get("Humidity_pct", "--")
gas       = rec.get("Gas_AQI", "--")
light     = rec.get("Light_Lux", "--")
motion    = rec.get("Motion_Detected", "--")
state_lbl = rec.get("state_label", "Normal")
fan_st    = rec.get("fan", "OFF")
alarm_st  = rec.get("alarm", "OFF")
servo_st  = rec.get("servo", "CLOSED")
buzzer_st = rec.get("buzzer", "OFF")
rgb_st    = rec.get("rgb_led", "GREEN")
anomaly   = rec.get("is_anomaly", False)
gas_pred  = rec.get("gas_pred", 0)
temp_pred = rec.get("temp_pred", 0)
reward    = rec.get("reward", 0)
attention = rec.get("attention", None)

pipeline_result = run_pipeline(rec, scaler=None, verbose=False)
attention = pipeline_result["_meta"]["attention"]

state_color = {"Normal": "#00e5a0", "Warning": "#f5b731", "Dangerous": "#ff4455"}.get(state_lbl, "#00e5a0")
state_pill_cls = {"Normal": "state-pill-normal", "Warning": "state-pill-warning", "Dangerous": "state-pill-danger"}.get(state_lbl, "state-pill-normal")

# =============================================================================
#  TOPBAR
# =============================================================================
st.markdown(f"""
<div class="alg-topbar">
    <div>
        <div class="alg-title">
            <span class="live-dot" style="background:{state_color};"></span>
            🧪 Adaptive Lab Guardian
        </div>
        <div class="alg-sub">Smart Adaptive Environment Monitoring &amp; Decision System</div>
    </div>
    <div class="alg-topbar-right">
        <div class="live-badge">
            <span class="live-dot"></span> LIVE
        </div>
        <div class="time-pill">📅 {now}</div>
        <div class="live-badge {state_pill_cls}" style="border-color:{state_color};color:{state_color};background:rgba(0,0,0,0.2);">
            {state_lbl.upper()}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
#  A — SENSOR READINGS
# =============================================================================
st.markdown('<div class="sec-hdr">A. Sensor Readings</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    def fmt_val(v, unit):
        try:
            return "YES" if (unit == "" and int(v) == 1) else ("NO" if unit == "" else f"{float(v):.1f}")
        except:
            return str(v)

    sensors = [
        (c1, "🌡️", "Temperature", temp, "°C",
         "background:rgba(255,68,85,0.12);border:1px solid rgba(255,68,85,0.2);"),
        (c2, "💧", "Humidity", humidity, "%",
         "background:rgba(77,159,255,0.12);border:1px solid rgba(77,159,255,0.2);"),
        (c3, "☁️", "Gas AQI", gas, "AQI",
         "background:rgba(245,183,49,0.12);border:1px solid rgba(245,183,49,0.2);"),
        (c4, "💡", "Light", light, "Lux",
         "background:rgba(245,183,49,0.12);border:1px solid rgba(245,183,49,0.2);"),
        (c5, "🚶", "Motion Detected", motion, "",
         "background:rgba(0,229,160,0.12);border:1px solid rgba(0,229,160,0.2);"),
    ]
    motion_color = "color:var(--green);" if (str(motion) == "1") else "color:var(--text3);"

    for col, icon, label, val, unit, icon_style in sensors:
        with col:
            display = fmt_val(val, unit)
            extra_style = motion_color if unit == "" else ""
            st.markdown(f"""
            <div class="s-card">
                <div class="s-icon-wrap" style="{icon_style}">{icon}</div>
                <div class="s-label">{label}</div>
                <div class="s-value" style="{extra_style}">{display}</div>
                <div class="s-unit">{unit if unit else "&nbsp;"}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  B — AI PIPELINE OUTPUT
# =============================================================================
st.markdown('<div class="sec-hdr">B. AI Pipeline Output</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    r1c1, r1c2, r1c3, r1c4 = st.columns([1.5, 1.5, 0.9, 0.9])

    state_val_cls = {"Normal": "p-val-green", "Warning": "p-val-yellow", "Dangerous": "p-val-red"}.get(state_lbl, "p-val-green")
    state_acc_cls = {"Normal": "p-card-accent-green", "Warning": "p-card-accent-yellow", "Dangerous": "p-card-accent-red"}.get(state_lbl, "p-card-accent-green")
    state_badge = badge(state_lbl)
    anomaly_badge_html = badge(str(anomaly))
    anomaly_acc = "p-card-accent-red" if anomaly else "p-card-accent-green"
    anomaly_lbl = "⚠ NEW PATTERN" if anomaly else "✔ Pattern Known"
    anomaly_cls = "p-val-red" if anomaly else "p-val-green"

    with r1c1:
        st.markdown(f"""
        <div class="p-card {state_acc_cls}" style="margin-bottom:0.7rem;">
            <div class="p-lbl">SOM — System State</div>
            <div class="{state_val_cls}">{state_lbl}</div>
            {state_badge}
        </div>
        <div class="p-card {anomaly_acc}">
            <div class="p-lbl">ART2 — Anomaly Detection</div>
            <div class="{anomaly_cls}" style="font-family:var(--display);font-size:1.1rem;font-weight:800;">{anomaly_lbl}</div>
            {anomaly_badge_html}
        </div>""", unsafe_allow_html=True)

    with r1c2:
        for lbl, val, cls, unit in [
            ("🔬 RBF — Gas Level Prediction", f"{float(gas_pred):.4f}", "p-val-blue", "normalised 0–1"),
            ("📈 RBF — Temperature Trend",    f"{float(temp_pred):.4f}", "p-val-blue", "normalised 0–1"),
            ("🎯 RL — Last Reward Signal",    f"{float(reward):+.2f}",  "p-val-yellow" if float(reward) >= 0 else "p-val-red", ""),
        ]:
            st.markdown(f"""
            <div class="p-card p-card-accent-blue" style="margin-bottom:0.7rem;">
                <div class="p-lbl">{lbl}</div>
                <div class="{cls}" style="font-family:var(--display);font-size:1.3rem;font-weight:800;">
                    {val} <span style="font-size:0.6rem;color:var(--text2);font-family:var(--mono);">{unit}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    total    = len(df)
    n_danger = int((df.get("state_label", pd.Series()) == "Dangerous").sum()) if "state_label" in df else 0
    n_anom   = int(df.get("is_anomaly", pd.Series(False)).sum()) if "is_anomaly" in df else 0

    with r1c3:
        st.markdown(f"""
        <div class="p-card p-card-accent-blue">
            <div class="p-lbl">Total Records</div>
            <div class="p-stat-num" style="color:var(--blue);">{total}</div>
            <div style="font-size:0.6rem;color:var(--text2);margin-top:0.3rem;">records logged</div>
        </div>""", unsafe_allow_html=True)

    with r1c4:
        st.markdown(f"""
        <div class="p-card" style="display:flex;flex-direction:column;gap:0.8rem;">
            <div>
                <div class="p-lbl" style="display:flex;align-items:center;gap:4px;">
                    <span style="color:var(--red);">⚠</span> Dangerous
                </div>
                <div class="p-stat-num" style="color:var(--red);font-size:1.8rem;">{n_danger}</div>
            </div>
            <div>
                <div class="p-lbl" style="display:flex;align-items:center;gap:4px;">
                    <span style="color:var(--yellow);">!</span> Anomalies
                </div>
                <div class="p-stat-num" style="color:var(--yellow);font-size:1.8rem;">{n_anom}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  C — ACTUATOR COMMANDS
# =============================================================================
st.markdown('<div class="sec-hdr">C. Final Actuator Commands</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns(5)

    actuators = [
        (a1, "🌀", "Fan",     fan_st),
        (a2, "🚨", "Alarm",   alarm_st),
        (a3, "🔧", "Servo",   servo_st),
        (a4, "🔔", "Buzzer",  buzzer_st),
        (a5, "💡", "RGB LED", rgb_st),
    ]

    state_color_map = {
        "ON":     "var(--green)",
        "OFF":    "var(--text3)",
        "OPEN":   "var(--green)",
        "CLOSED": "var(--text3)",
        "RED":    "var(--red)",
        "YELLOW": "var(--yellow)",
        "GREEN":  "var(--green)",
    }

    for col, icon, name, state in actuators:
        with col:
            s = str(state).upper()
            if name == "RGB LED":
                state_html = badge(s)
            else:
                color = state_color_map.get(s, "var(--text2)")
                state_html = f'<div class="act-state" style="color:{color};">{s}</div>'
            st.markdown(f"""
            <div class="act-card">
                <div class="act-icon">{icon}</div>
                <div class="act-label">{name}</div>
                {state_html}
            </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  D — TIME-SERIES CHARTS
# =============================================================================
st.markdown('<div class="sec-hdr">D. Real-time Trends</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)
    cl, cr = st.columns(2)
    cl2, cr2 = st.columns(2)

    chart_configs = [
        (cl,  "Temp_C",       "Temperature (°C)",  "#ff4455"),
        (cr,  "Gas_AQI",      "Gas AQI",           "#f5b731"),
        (cl2, "Humidity_pct", "Humidity (%)",      "#4d9fff"),
        (cr2, "Light_Lux",    "Light (Lux)",       "#00e5a0"),
    ]

    for col, field, title, color in chart_configs:
        with col:
            st.markdown(f'<div class="ch-card"><div class="ch-title" style="color:{color};">{title}</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({title: df[field].reset_index(drop=True)})
            st.area_chart(chart_df, height=130, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  E — GNN ATTENTION GRAPH
# =============================================================================

st.markdown('<div class="sec-hdr">E. GNN Attention Graph</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if attention is not None:
        fig = draw_attention_graph(attention)
        if fig:
            st.pyplot(fig, use_container_width=False)
        else:
            st.warning("Failed to draw attention graph")
    else:
        st.warning("GNN attention not available")

# =============================================================================
#  F — EVENT LOG
# =============================================================================
st.markdown('<div class="sec-hdr">E. Event Log (Recent)</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="content-pad">', unsafe_allow_html=True)

    cols_needed = ["timestamp", "Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux",
                   "Motion_Detected", "state_label", "fan", "alarm"]
    log_df = df[[c for c in cols_needed if c in df.columns]].tail(10).iloc[::-1].copy()

    # Build the full HTML string in one shot to avoid Streamlit f-string escaping
    ev_parts = ["""<div class="ev-table"><div class="ev-hdr">
        <span>Time</span>
        <span>Sensors (T / H / G / L / M)</span>
        <span>State</span>
        <span>Actions</span>
    </div>"""]

    for _, row in log_df.iterrows():
        ts    = row.get("timestamp", "—")
        t     = row.get("Temp_C", "—")
        h     = row.get("Humidity_pct", "—")
        g     = row.get("Gas_AQI", "—")
        lx    = row.get("Light_Lux", "—")
        m     = row.get("Motion_Detected", "—")
        st_r  = str(row.get("state_label", "—"))
        fan_r = str(row.get("fan", "—"))
        alm_r = str(row.get("alarm", "—"))
        try:
            mot_str = "YES" if int(m) == 1 else "NO"
            summary = f"{float(t):.1f} / {float(h):.1f} / {float(g):.0f} / {float(lx):.0f} / {mot_str}"
        except Exception:
            summary = "—"

        st_badge  = badge(st_r)
        fan_badge = badge(fan_r)
        alm_badge = badge(alm_r)

        ev_parts.append(
            '<div class="ev-row">'
            f'<span class="ev-time">{ts}</span>'
            f'<span style="font-size:0.65rem;">{summary}</span>'
            f'<span>{st_badge}</span>'
            f'<span style="font-size:0.64rem;">Fan {fan_badge} &nbsp;Alm {alm_badge}</span>'
            '</div>'
        )

    ev_parts.append("</div>")
    st.markdown("".join(ev_parts), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
#  FOOTER
# =============================================================================
st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;border-top:1px solid var(--border);
            margin-top:1.5rem;font-size:0.6rem;color:var(--text3);">
    © 2026 Adaptive Lab Guardian. All rights reserved.
</div>""", unsafe_allow_html=True)

# =============================================================================
#  AUTO-REFRESH
# =============================================================================
if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()