import streamlit as st
import time
import os
import json
import pandas as pd

st.set_page_config(page_title="Defence Surveillance System", layout="wide")
st.title("🚨 Defence Surveillance System")

st.markdown("""
---
**Live monitoring with real-time detection, alerts, and forensic logs.**
""")

def read_state():
    path = os.path.join("logs", "state.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def read_history_tail(n=15):
    csvp = os.path.join("logs", "detections.csv")
    if os.path.exists(csvp):
        try:
            df = pd.read_csv(csvp)
            return df.tail(n)
        except Exception:
            return None
    return None

# Sidebar controls
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    refresh_rate = st.slider("Refresh rate (seconds)", 1, 5, 2)
    show_history = st.checkbox("Show detection history", value=True)
    show_alerts = st.checkbox("Show alert folder", value=True)

# Main loop
state_metric = st.empty()
stream_box = st.empty()
history_box = st.empty()
alerts_box = st.empty()

while True:
    s = read_state()
    person_count = s.get("person_count", 0)
    alerts = s.get("alerts", [])
    last_alert = alerts[-1] if alerts else "No recent alerts"
    ts = s.get("timestamp", "-")
    
    # Metrics row
    with state_metric.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Persons", person_count, delta=None)
        status_color = "🔴 ALERT" if alerts else "🟢 SAFE"
        c2.metric("Status", status_color)
        c3.write(f"**⏰ Time:** {ts}")
        c4.write(f"**📬 Last Alert:**  \n{last_alert[:50]}..." if len(str(last_alert)) > 50 else f"**📬 Last Alert:**  \n{last_alert}")
    
    # Live stream from Flask MJPEG server
    with stream_box.container():
        st.markdown("### 🎥 Live Camera Feed")
        try:
            st.markdown(
                f'<img src="http://localhost:5000/video_feed" width="100%" style="border-radius: 8px; border: 2px solid #ccc;">',
                unsafe_allow_html=True
            )
            st.caption("MJPEG stream (20 fps) — served by Flask stream_server.py")
        except Exception as e:
            st.warning(f"⚠️ Stream not available: {e}  \nEnsure `stream_server.py` is running on port 5000.")
    
    # Alert history
    if show_history:
        with history_box.container():
            st.markdown("### 📊 Recent Detections (CSV Log)")
            df = read_history_tail(15)
            if df is not None:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No detections logged yet.")
    
    # Alert screenshots folder
    if show_alerts:
        with alerts_box.container():
            st.markdown("### 🚨 Recent Alert Screenshots")
            alerts_dir = os.path.join("logs", "alerts")
            if os.path.exists(alerts_dir):
                alert_files = sorted(
                    [f for f in os.listdir(alerts_dir) if f.endswith(".jpg")],
                    reverse=True
                )[:5]
                if alert_files:
                    cols = st.columns(min(3, len(alert_files)))
                    for i, fname in enumerate(alert_files):
                        with cols[i % len(cols)]:
                            img_path = os.path.join(alerts_dir, fname)
                            st.image(img_path, caption=fname, use_column_width=True)
                else:
                    st.info("No alert screenshots yet.")
            else:
                st.info("Alert folder not yet created.")
    
    time.sleep(refresh_rate)
