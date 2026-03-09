"""
Main Entry Point
=================
AI-Based Real-Time Intrusion Detection System for Border & Base Surveillance
with Face Recognition for Authorized/Unauthorized Person Detection

Pipeline: Camera → Frame Processing → YOLOv8 Detection → Face Recognition → Display

Usage:                                          
    python main.py
"""

import cv2
from ultralytics import YOLO
from video_capture import VideoCapture
from detection import detect_persons, draw_detections, detect_objects, draw_object_detections
from rules import RuleEngine
from alerts import AlertManager
from logger import DetectionLogger
from face_rec import FaceRecognizer
import os
import datetime
import json
import time

def main():
    # Load YOLOv8 model
    print("[Main] Loading YOLOv8 model …")
    model = YOLO("yolov8n.pt")

    # Initialize face recognizer (reads from Authorized_persons/ in parent dir)
    print("[Main] Loading face recognition …")
    face_rec = FaceRecognizer()

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

            # Detect persons (YOLO)
            detections = detect_persons(model, frame)

            # Detect suspicious/dangerous objects (same YOLO model)
            object_detections = detect_objects(model, frame)
            high_threat_objects = [o for o in object_detections if o["threat_level"] == "HIGH"]

            # Face recognition
            face_results = face_rec.recognize(frame)
            unauthorized_faces = [f for f in face_results if not f["authorized"]]
            authorized_faces = [f for f in face_results if f["authorized"]]

            # Draw YOLO bounding boxes
            annotated = draw_detections(frame, detections)

            # Draw face recognition boxes (green=authorized, red=unauthorized)
            annotated = face_rec.draw_face_results(annotated, face_results)

            # Draw threat object boxes (red/orange/yellow by threat level)
            annotated = draw_object_detections(annotated, object_detections)

            # Show counts on frame
            obj_warn = f" | Threats: {len(object_detections)}" if object_detections else ""
            cv2.putText(
                annotated,
                f"Persons: {len(detections)} | Auth: {len(authorized_faces)} | Unauth: {len(unauthorized_faces)}{obj_warn}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Evaluate rules (now includes face recognition + object detection results)
            frame_alerts = rules.update(detections, frame, timestamp=ts,
                                        face_results=face_results,
                                        object_detections=object_detections)
            # draw alerts on frame
            y = 50
            for alert in frame_alerts[:6]:
                cv2.putText(annotated, alert, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y += 22

            if frame_alerts:
                # Print and dispatch alerts
                alerts.send_alerts(frame_alerts)

                # Save alert screenshot
                try:
                    alerts_dir = os.path.join("logs", "alerts")
                    os.makedirs(alerts_dir, exist_ok=True)
                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    alert_fname = f"alert_{ts_str}.jpg"
                    alert_path = os.path.join(alerts_dir, alert_fname)
                    cv2.imwrite(alert_path, annotated.copy())
                except Exception:
                    alert_path = ""

                # Save log entry
                try:
                    dlogger.log(len(detections), frame=annotated.copy(), alerts=frame_alerts)
                except Exception:
                    pass

                # Append to alert history JSON for dashboard feed
                try:
                    history_path = os.path.join("logs", "alert_history.json")
                    history = []
                    if os.path.exists(history_path):
                        with open(history_path, "r", encoding="utf-8") as hf:
                            history = json.load(hf)
                    for fa in frame_alerts:
                        history.append({
                            "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
                            "alert": fa,
                            "person_count": len(detections),
                            "unauthorized_count": len(unauthorized_faces),
                            "image": alert_path,
                        })
                    history = history[-50:]
                    with open(history_path, "w", encoding="utf-8") as hf:
                        json.dump(history, hf)
                except Exception:
                    pass

            cv2.imshow("Intrusion Detection System - press 'q' to quit", annotated)

            # Always update state for dashboard (even when no alerts)
            try:
                auth_names = [f["name"] for f in authorized_faces]
                obj_names = [o["class_name"] for o in object_detections]
                state = {
                    "timestamp": datetime.datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "person_count": len(detections),
                    "alerts": frame_alerts,
                    "authorized_count": len(authorized_faces),
                    "unauthorized_count": len(unauthorized_faces),
                    "authorized_names": auth_names,
                    "total_authorized_db": face_rec.num_authorized,
                    "threat_objects": obj_names,
                    "threat_count": len(object_detections),
                    "high_threat_count": len(high_threat_objects),
                    "last_alert_image": "",
                }
                with open(os.path.join("logs", "state.json"), "w", encoding="utf-8") as sf:
                    json.dump(state, sf)
            except Exception:
                pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        print("[Main] System shut down.")

if __name__ == "__main__":
    main()
