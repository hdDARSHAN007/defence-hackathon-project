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
    /* Overall dark military theme */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #111927 50%, #0d1520 100%);
    }
    /* Header banner */
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
    /* Status cards */
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
    /* Section headers */
    .section-header {
        color: #00d4ff;
        font-size: 1.15em;
        font-weight: 600;
        padding: 10px 0 8px 0;
        border-bottom: 1px solid #2a3a4a;
        margin-bottom: 12px;
        letter-spacing: 1px;
    }
    /* Alert feed item */
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
    /* Live feed container */
    .live-feed-box {
        border: 2px solid #2d4a5e;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 25px rgba(0,150,255,0.15);
        position: relative;
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
    /* Captured moment card */
    .moment-card {
        background: #141e2b;
        border: 1px solid #2a3a4a;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .moment-card img { width: 100%; }
    .moment-info {
        padding: 8px 12px;
        color: #aaa;
        font-size: 0.8em;
    }
    .moment-info .moment-alert {
        color: #ff6b6b;
        font-weight: 600;
        font-size: 0.9em;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1520 0%, #111927 100%) !important;
        border-right: 1px solid #2a3a4a;
    }
    /* Table styling */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ─────────────────────────────────────────────────
LOG_DIR = "logs"
ALERTS_DIR = os.path.join(LOG_DIR, "alerts")

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
    """Read the alert history JSON file (list of recent alert events)."""
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
    """Return the latest n alert screenshot paths, newest first."""
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

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Command Center")
    st.markdown("---")
    refresh_rate = st.slider("⏱️ Refresh Rate (sec)", 1, 5, 2)
    st.markdown("---")
    st.markdown("### 📋 Display Options")
    show_live = st.checkbox("Live Camera Feed", value=True)
    show_alerts_feed = st.checkbox("Live Alert Feed", value=True)
    show_captures = st.checkbox("Captured Moments", value=True)
    show_history = st.checkbox("Detection Log Table", value=True)
    num_captures = st.slider("Captures to show", 3, 12, 6)
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("[📹 Direct Stream](http://localhost:5000)")
    st.markdown("---")
    st.caption("AI-Based Real-Time Intrusion Detection System for Border & Base Surveillance")

# ── Placeholders ─────────────────────────────────────────────────────
header_ph = st.empty()
metrics_ph = st.empty()
col_ph = st.empty()
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

    # ── Header Banner ────────────────────────────────────────────────
    with header_ph.container():
        st.markdown("""
        <div class="header-banner">
            <h1>🛡️ DEFENCE SURVEILLANCE COMMAND CENTER</h1>
            <p>AI-Powered Real-Time Intrusion Detection &bull; Border &amp; Base Security</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Status Cards Row ─────────────────────────────────────────────
    with metrics_ph.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            val_class = "val-red" if person_count > 0 else "val-green"
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Persons Detected</div>
                <div class="value {val_class}">{person_count}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            if is_alert:
                st.markdown("""
                <div class="status-card">
                    <div class="label">System Status</div>
                    <div class="value val-red">⚠ ALERT</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-card">
                    <div class="label">System Status</div>
                    <div class="value val-green">✓ SECURE</div>
                </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Total Alerts</div>
                <div class="value val-yellow">{total_alerts}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Last Update</div>
                <div class="value val-blue" style="font-size:1em;">{ts}</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            active_rules = sum([
                person_count > 1,
                any("Restricted" in a for a in current_alerts),
                any("Loitering" in a for a in current_alerts),
                any("Night" in a for a in current_alerts),
            ])
            st.markdown(f"""
            <div class="status-card">
                <div class="label">Active Rules Triggered</div>
                <div class="value val-yellow">{active_rules} / 4</div>
            </div>""", unsafe_allow_html=True)

    # ── Main Content: Live Feed + Alert Feed ─────────────────────────
    with col_ph.container():
        left_col, right_col = st.columns([3, 2])

        # Live camera feed
        with left_col:
            if show_live:
                st.markdown('<div class="section-header"><span class="live-badge">● LIVE</span> Camera Feed</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="live-feed-box">
                    <img src="http://localhost:5000/video_feed" width="100%" style="display:block;">
                </div>
                """, unsafe_allow_html=True)
                st.caption("MJPEG stream (20 fps) — ensure stream_server.py is running")

        # Live alert feed
        with right_col:
            if show_alerts_feed:
                st.markdown('<div class="section-header">🚨 Live Alert Feed</div>', unsafe_allow_html=True)

                # Show current active alerts
                if current_alerts:
                    for alert_msg in current_alerts:
                        st.markdown(f"""
                        <div class="alert-item">
                            <div class="alert-time">🕐 {ts}</div>
                            <div class="alert-msg">⚠ {alert_msg}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align:center; padding:40px 20px; color:#4a6a7a;">
                        <div style="font-size:3em;">✓</div>
                        <div style="font-size:1.1em; color:#00ff88;">All Clear — No Active Threats</div>
                        <div style="font-size:0.8em; color:#4a6a7a; margin-top:8px;">System monitoring...</div>
                    </div>""", unsafe_allow_html=True)

                # Show alert history from JSON
                alert_history = read_alert_log()
                if alert_history:
                    st.markdown('<div class="section-header" style="margin-top:16px;">📜 Recent Alert History</div>', unsafe_allow_html=True)
                    for entry in alert_history[-8:]:
                        ets = entry.get("timestamp", "")
                        emsg = entry.get("alert", "")
                        st.markdown(f"""
                        <div class="alert-item" style="border-left-color:#ff8800;">
                            <div class="alert-time">🕐 {ets}</div>
                            <div class="alert-msg" style="color:#ffaa44;">📌 {emsg}</div>
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
                        # Parse timestamp from filename: alert_YYYYMMDD_HHMMSS_mmm.jpg
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
