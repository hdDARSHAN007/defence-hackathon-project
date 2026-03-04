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


def main():
    # Load YOLOv8 model
    print("[Main] Loading YOLOv8 model …")
    model = YOLO("yolov8n.pt")

    # Initialize camera
    cam = VideoCapture(source=0, width=640, height=480)
    cam.start()

    try:
        for frame in cam.frames():
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
