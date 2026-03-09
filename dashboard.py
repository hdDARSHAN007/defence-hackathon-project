import streamlit as st
import time
import os
import json
import glob
import pandas as pd
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Defence Surveillance Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for military-style dark theme ─────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #111927 50%, #0d1520 100%);
    }
    .header-banner {
        background: linear-gradient(90deg, #1a2332 0%, #243447 50%, #1a2332 100%);
        border: 1px solid #2d4a5e;
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,150,255,0.1);
    }
    .header-banner h1 {
        color: #00d4ff;
        font-size: 2.2em;
        margin: 0;
        text-shadow: 0 0 20px rgba(0,212,255,0.3);
        letter-spacing: 3px;
    }
    .header-banner p {
        color: #7aa2b8;
        margin: 5px 0 0 0;
        font-size: 0.95em;
        letter-spacing: 1px;
    }
    .status-card {
        background: linear-gradient(145deg, #141e2b, #1a2a3a);
        border: 1px solid #2a3a4a;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .status-card:hover { transform: translateY(-2px); }
    .status-card .label {
        color: #6b8fa8;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .status-card .value {
        font-size: 2em;
        font-weight: 700;
        margin: 4px 0;
    }
    .val-green { color: #00ff88; text-shadow: 0 0 10px rgba(0,255,136,0.3); }
    .val-red { color: #ff4444; text-shadow: 0 0 10px rgba(255,68,68,0.4); animation: pulse-red 1.5s infinite; }
    .val-blue { color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.3); }
    .val-yellow { color: #ffcc00; text-shadow: 0 0 10px rgba(255,204,0,0.3); }
    @keyframes pulse-red {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .section-header {
        color: #00d4ff;
        font-size: 1.15em;
        font-weight: 600;
        padding: 10px 0 8px 0;
        border-bottom: 1px solid #2a3a4a;
        margin-bottom: 12px;
        letter-spacing: 1px;
    }
    .alert-item {
        background: linear-gradient(90deg, #1c1a0e, #1a1a1a);
        border-left: 4px solid #ff4444;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9em;
    }
    .alert-item .alert-time { color: #888; font-size: 0.8em; }
    .alert-item .alert-msg { color: #ffcc00; font-weight: 500; }
    .live-feed-box {
        border: 2px solid #2d4a5e;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 25px rgba(0,150,255,0.15);
    }
    .live-badge {
        display: inline-block;
        background: #ff0000;
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75em;
        font-weight: 700;
        letter-spacing: 1px;
        animation: pulse-red 1.5s infinite;
        margin-right: 8px;
    }
    .face-card {
        background: linear-gradient(145deg, #0d1a0d, #1a2a1a);
        border: 1px solid #2a4a2a;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .face-card.unauthorized {
        background: linear-gradient(145deg, #1a0d0d, #2a1a1a);
        border: 1px solid #4a2a2a;
    }
    .threat-card {
        background: linear-gradient(145deg, #1a1a0d, #2a2a1a);
        border: 1px solid #4a4a2a;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .threat-card.high {
        background: linear-gradient(145deg, #1a0d0d, #2a1a1a);
        border: 1px solid #4a2a2a;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520 0%, #111927 100%) !important;
        border-right: 1px solid #2a3a4a;
    }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ─────────────────────────────────────────────────
LOG_DIR = "logs"
ALERTS_DIR = os.path.join(LOG_DIR, "alerts")

# Find Authorized_persons in parent dir
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "Authorized_persons")

def read_state():
    path = os.path.join(LOG_DIR, "state.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def read_alert_log():
    path = os.path.join(LOG_DIR, "alert_history.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def read_history_tail(n=20):
    csvp = os.path.join(LOG_DIR, "detections.csv")
    if os.path.exists(csvp):
        try:
            df = pd.read_csv(csvp)
            return df.tail(n)
        except Exception:
            return None
    return None

def get_alert_images(n=12):
    if not os.path.exists(ALERTS_DIR):
        return []
    files = sorted(
        [f for f in os.listdir(ALERTS_DIR) if f.lower().endswith(".jpg")],
        reverse=True,
    )
    return files[:n]

def get_total_alerts():
    if not os.path.exists(ALERTS_DIR):
        return 0
    return len([f for f in os.listdir(ALERTS_DIR) if f.lower().endswith(".jpg")])

def get_authorized_people():
    if not os.path.exists(AUTH_DIR):
        return []
    return [f for f in os.listdir(AUTH_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Command Center")
    st.markdown("---")
    refresh_rate = st.slider("⏱️ Refresh Rate (sec)", 1, 5, 2)
    st.markdown("---")
    st.markdown("### 📋 Display Options")
    show_live = st.checkbox("Live Camera Feed", value=True)
    show_face_panel = st.checkbox("Face Recognition Panel", value=True)
    show_threat_panel = st.checkbox("Threat Detection Panel", value=True)
    show_alerts_feed = st.checkbox("Live Alert Feed", value=True)
    show_captures = st.checkbox("Captured Moments", value=True)
    show_history = st.checkbox("Detection Log Table", value=True)
    num_captures = st.slider("Captures to show", 3, 12, 6)
    st.markdown("---")

    # Show authorized personnel in sidebar
    st.markdown("### 👤 Authorized Personnel")
    auth_people = get_authorized_people()
    if auth_people:
        import re
        seen_names = set()
        for p in auth_people:
            base = os.path.splitext(p)[0]
            name = re.sub(r'\d+$', '', base).replace("_", " ").strip().title()
            if name not in seen_names:
                st.markdown(f"✅ **{name}**")
                seen_names.add(name)
        st.caption(f"{len(auth_people)} training photos loaded")
    else:
        st.warning("No authorized faces. Add photos to Authorized_persons/")

    st.markdown("---")
    st.markdown("[📹 Direct Stream](http://localhost:5000)")
    st.caption("AI-Based Intrusion Detection with Face Recognition")

# ── Placeholders ─────────────────────────────────────────────────────
header_ph = st.empty()
metrics_ph = st.empty()
col_ph = st.empty()
face_ph = st.empty()
threat_ph = st.empty()
captures_ph = st.empty()
history_ph = st.empty()

# ── Main refresh loop ────────────────────────────────────────────────
while True:
    state = read_state()
    person_count = state.get("person_count", 0)
    current_alerts = state.get("alerts", [])
    ts = state.get("timestamp", "-")
    total_alerts = get_total_alerts()
    is_alert = len(current_alerts) > 0
    auth_count = state.get("authorized_count", 0)
    unauth_count = state.get("unauthorized_count", 0)
    auth_names = state.get("authorized_names", [])
    total_auth_db = state.get("total_authorized_db", 0)
    threat_objects = state.get("threat_objects", [])
    threat_count = state.get("threat_count", 0)
    high_threat_count = state.get("high_threat_count", 0)

    # ── Header Banner ────────────────────────────────────────────────
    with header_ph.container():
        st.markdown("""
        <div class="header-banner">
            <h1>🛡️ DEFENCE SURVEILLANCE COMMAND CENTER</h1>
            <p>AI-Powered Intrusion Detection &bull; Face Recognition &bull; Border &amp; Base Security</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Status Cards Row ─────────────────────────────────────────────
    with metrics_ph.container():
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1:
            val_class = "val-red" if person_count > 0 else "val-green"
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Persons</div>
                <div class="value {val_class}">{person_count}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            if is_alert:
                st.markdown("""
                <div class="status-card">
                    <div class="label">Status</div>
                    <div class="value val-red">⚠ ALERT</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card">
                    <div class="label">Status</div>
                    <div class="value val-green">✓ SECURE</div>
                </div>""", unsafe_allow_html=True)
        with c3:
            vc = "val-green" if auth_count > 0 else "val-blue"
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Authorized</div>
                <div class="value {vc}">{auth_count}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            uc = "val-red" if unauth_count > 0 else "val-green"
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Unauthorized</div>
                <div class="value {uc}">{unauth_count}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            tc = "val-red" if high_threat_count > 0 else ("val-yellow" if threat_count > 0 else "val-green")
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Threats</div>
                <div class="value {tc}">{threat_count}</div>
            </div>""", unsafe_allow_html=True)
        with c6:
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Total Alerts</div>
                <div class="value val-yellow">{total_alerts}</div>
            </div>""", unsafe_allow_html=True)
        with c7:
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Last Update</div>
                <div class="value val-blue" style="font-size:0.85em;">{ts}</div>
            </div>""", unsafe_allow_html=True)

    # ── Main Content: Live Feed + Alert Feed ─────────────────────────
    with col_ph.container():
        left_col, right_col = st.columns([3, 2])

        with left_col:
            if show_live:
                st.markdown('<div class="section-header"><span class="live-badge">● LIVE</span> Camera Feed</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="live-feed-box">
                    <img src="http://localhost:5000/video_feed" width="100%" style="display:block;">
                </div>
                """, unsafe_allow_html=True)

        with right_col:
            if show_alerts_feed:
                st.markdown('<div class="section-header">🚨 Live Alert Feed</div>', unsafe_allow_html=True)
                if current_alerts:
                    for alert_msg in current_alerts:
                        border_color = "#ff0000" if "INTRUDER" in alert_msg else "#ff4444"
                        st.markdown(f"""
                        <div class="alert-item" style="border-left-color:{border_color};">
                            <div class="alert-time">🕐 {ts}</div>
                            <div class="alert-msg">⚠ {alert_msg}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align:center; padding:40px 20px; color:#4a6a7a;">
                        <div style="font-size:3em;">✓</div>
                        <div style="font-size:1.1em; color:#00ff88;">All Clear — No Active Threats</div>
                        <div style="font-size:0.8em; color:#4a6a7a; margin-top:8px;">System monitoring...</div>
                    </div>""", unsafe_allow_html=True)

                alert_history = read_alert_log()
                if alert_history:
                    st.markdown('<div class="section-header" style="margin-top:16px;">📜 Recent History</div>', unsafe_allow_html=True)
                    for entry in alert_history[-6:]:
                        ets = entry.get("timestamp", "")
                        emsg = entry.get("alert", "")
                        st.markdown(f"""
                        <div class="alert-item" style="border-left-color:#ff8800;">
                            <div class="alert-time">🕐 {ets}</div>
                            <div class="alert-msg" style="color:#ffaa44;">📌 {emsg}</div>
                        </div>""", unsafe_allow_html=True)

    # ── Face Recognition Panel ───────────────────────────────────────
    if show_face_panel:
        with face_ph.container():
            st.markdown('<div class="section-header">🧑 Face Recognition Status</div>', unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.markdown(f"""
                <div class="face-card">
                    <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Database</div>
                    <div style="color:#00d4ff; font-size:1.8em; font-weight:700;">{total_auth_db} persons</div>
                    <div style="color:#555; font-size:0.8em;">registered in Authorized_persons/</div>
                </div>""", unsafe_allow_html=True)
            with fc2:
                if auth_names:
                    names_html = "".join([f'<div style="color:#00ff88;">✅ {n}</div>' for n in auth_names])
                    st.markdown(f"""
                    <div class="face-card">
                        <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Currently Recognized</div>
                        {names_html}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="face-card">
                        <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Currently Recognized</div>
                        <div style="color:#555;">No authorized faces in view</div>
                    </div>""", unsafe_allow_html=True)
            with fc3:
                if unauth_count > 0:
                    st.markdown(f"""
                    <div class="face-card unauthorized">
                        <div style="color:#ff4444; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">⚠ INTRUDER ALERT</div>
                        <div style="color:#ff4444; font-size:1.8em; font-weight:700;">{unauth_count} unknown</div>
                        <div style="color:#ff6b6b; font-size:0.85em;">Unauthorized person detected!</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="face-card">
                        <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Intruder Check</div>
                        <div style="color:#00ff88; font-size:1.4em; font-weight:700;">✓ Clear</div>
                        <div style="color:#555; font-size:0.85em;">No unauthorized persons</div>
                    </div>""", unsafe_allow_html=True)

    # ── Threat Detection Panel ───────────────────────────────────────
    if show_threat_panel:
        with threat_ph.container():
            st.markdown('<div class="section-header">🔫 Threat Object Detection</div>', unsafe_allow_html=True)
            tc1, tc2 = st.columns(2)
            with tc1:
                if high_threat_count > 0:
                    st.markdown(f"""
                    <div class="threat-card high">
                        <div style="color:#ff4444; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">⚠ WEAPON ALERT</div>
                        <div style="color:#ff4444; font-size:1.8em; font-weight:700;">{high_threat_count} weapon(s)</div>
                        <div style="color:#ff6b6b; font-size:0.85em;">Dangerous object in view!</div>
                    </div>""", unsafe_allow_html=True)
                elif threat_count > 0:
                    st.markdown(f"""
                    <div class="threat-card">
                        <div style="color:#ffcc00; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Suspicious Objects</div>
                        <div style="color:#ffcc00; font-size:1.8em; font-weight:700;">{threat_count} detected</div>
                        <div style="color:#aaa; font-size:0.85em;">Monitoring...</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="threat-card">
                        <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Threat Scan</div>
                        <div style="color:#00ff88; font-size:1.4em; font-weight:700;">✓ Clear</div>
                        <div style="color:#555; font-size:0.85em;">No weapons or suspicious objects</div>
                    </div>""", unsafe_allow_html=True)
            with tc2:
                if threat_objects:
                    items_html = ""
                    for obj_name in threat_objects:
                        icon = "🔪" if obj_name in ("Knife", "Scissors") else "🎒" if obj_name in ("Backpack", "Suitcase") else "📦"
                        items_html += f'<div style="color:#ffcc00;">{icon} {obj_name}</div>'
                    st.markdown(f"""
                    <div class="threat-card">
                        <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Objects Detected</div>
                        {items_html}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="threat-card">
                        <div style="color:#6b8fa8; font-size:0.8em; text-transform:uppercase; letter-spacing:1px;">Objects Detected</div>
                        <div style="color:#555;">None</div>
                    </div>""", unsafe_allow_html=True)

    # ── Captured Moments Gallery ─────────────────────────────────────
    if show_captures:
        with captures_ph.container():
            st.markdown('<div class="section-header">📸 Captured Suspicious Moments</div>', unsafe_allow_html=True)
            alert_files = get_alert_images(num_captures)
            if alert_files:
                cols = st.columns(3)
                for i, fname in enumerate(alert_files):
                    with cols[i % 3]:
                        img_path = os.path.join(ALERTS_DIR, fname)
                        try:
                            parts = fname.replace("alert_", "").replace(".jpg", "")
                            dt_parts = parts.split("_")
                            date_str = dt_parts[0]
                            time_str = dt_parts[1] if len(dt_parts) > 1 else ""
                            display_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        except Exception:
                            display_time = fname
                        st.image(img_path, use_container_width=True)
                        st.caption(f"🕐 {display_time}")
            else:
                st.info("No captured moments yet. Alerts will appear here when suspicious activity is detected.")

    # ── Detection Log Table ──────────────────────────────────────────
    if show_history:
        with history_ph.container():
            st.markdown('<div class="section-header">📊 Detection Log</div>', unsafe_allow_html=True)
            df = read_history_tail(20)
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No detections logged yet.")

    time.sleep(refresh_rate)
