"""
Main Entry Point
=================
AI-Based Real-Time Intrusion Detection System for Border & Base Surveillance

Pipeline: Camera → Frame Processing → YOLOv8 Detection → Display

Usage:                                          
    python main.py
"""

import cv2
from ultralytics import YOLO
from video_capture import VideoCapture
from detection import detect_persons, draw_detections
from rules import RuleEngine
from alerts import AlertManager
from logger import DetectionLogger
import os
import datetime
import json
import time

def main():
    # Load YOLOv8 model
    print("[Main] Loading YOLOv8 model …")
    model = YOLO("yolov8n.pt")

    # Initialize camera
    cam = VideoCapture(source=0, width=640, height=480)
    cam.start()

    # Rules engine
    rules = RuleEngine(stay_seconds=10.0)

    # Alert manager (reads config from env)
    alerts = AlertManager(cooldown=8.0)

    # Detection logger (saves CSV + screenshots)
    dlogger = DetectionLogger(log_dir="logs")

    try:
        for frame in cam.frames():
            ts = time.time()
            # ensure logs dir exists and write latest frame for dashboard
            try:
                os.makedirs("logs", exist_ok=True)
                latest_path = os.path.join("logs", "latest.jpg")
                cv2.imwrite(latest_path, frame)
            except Exception:
                pass
            # Detect persons
            detections = detect_persons(model, frame)

            # Draw bounding boxes
            annotated = draw_detections(frame, detections)

            # Show person count
            cv2.putText(
                annotated,
                f"Persons detected: {len(detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Evaluate rules and show alerts
            frame_alerts = rules.update(detections, frame, timestamp=ts)
            # draw alerts on frame (top-left)
            y = 40
            for alert in frame_alerts[:6]:
                cv2.putText(annotated, alert, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 22
            if frame_alerts:
                # Print and dispatch alerts (console/sound/email/telegram/dashboard)
                alerts.send_alerts(frame_alerts)

                # Save a raw alert screenshot (important for forensics)
                try:
                    alerts_dir = os.path.join("logs", "alerts")
                    os.makedirs(alerts_dir, exist_ok=True)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    alert_fname = f"alert_{ts}.jpg"
                    alert_path = os.path.join(alerts_dir, alert_fname)
                    # save the raw annotated frame for context
                    cv2.imwrite(alert_path, annotated.copy())
                except Exception:
                    pass

                # Save a log entry and screenshot for the event (CSV + fullframe)
                try:
                    dlogger.log(len(detections), frame=annotated.copy(), alerts=frame_alerts)
                except Exception:
                    pass

                # Update state file for dashboard consumption
                try:
                    state = {
                        "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
                        "person_count": len(detections),
                        "alerts": frame_alerts,
                        "last_alert_image": alert_path if 'alert_path' in locals() else "",
                    }
                    with open(os.path.join("logs", "state.json"), "w", encoding="utf-8") as sf:
                        json.dump(state, sf)
                except Exception:
                    pass

            cv2.imshow("Intrusion Detection System - press 'q' to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        print("[Main] System shut down.")

if __name__ == "__main__":
    main()
