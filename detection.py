"""
AI Object Detection Module (YOLOv8)
====================================
Part of: AI-Based Real-Time Intrusion Detection System
Role   : Detects humans in preprocessed frames using YOLOv8 and draws
         bounding boxes around detected persons.

Usage:
    python detection.py
"""

import cv2
from ultralytics import YOLO
from video_capture import VideoCapture

# YOLO class ID for "person" in the COCO dataset
PERSON_CLASS_ID = 0

# Bounding box styling
BOX_COLOR = (0, 0, 255)       # red in BGR
BOX_THICKNESS = 2
LABEL_COLOR = (255, 255, 255) # white text
LABEL_BG = (0, 0, 255)        # red background for label


def detect_persons(model: YOLO, frame):
    """
    Run YOLOv8 inference on a single frame and return only 'person' detections.

    Parameters
    ----------
    model : YOLO
        Loaded YOLOv8 model.
    frame : np.ndarray
        Preprocessed BGR frame from VideoCapture.

    Returns
    -------
    list[dict]
        Each dict contains:
          - 'bbox'       : (x1, y1, x2, y2) pixel coordinates
          - 'confidence' : float 0-1
    """
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != PERSON_CLASS_ID:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
            })

    return detections


def draw_detections(frame, detections):
    """
    Draw bounding boxes and confidence labels on the frame.

    Parameters
    ----------
    frame : np.ndarray
        The frame to annotate (modified in-place).
    detections : list[dict]
        Output from detect_persons().

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = f"Person {conf:.0%}"

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), LABEL_BG, -1)

        # label text
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, LABEL_COLOR, 1, cv2.LINE_AA,
        )

    return frame


# ======================================================================
# Main loop — capture → detect → annotate → display
# ======================================================================
if __name__ == "__main__":
    # Load YOLOv8 pretrained model (downloads automatically on first run)
    print("[Detection] Loading YOLOv8 model …")
    model = YOLO("yolov8n.pt")  # 'n' = nano, fastest variant

    cam = VideoCapture(source=0, width=640, height=480)
    cam.start()

    try:
        for frame in cam.frames():
            detections = detect_persons(model, frame)
            annotated = draw_detections(frame, detections)

            # Show person count on frame
            count_label = f"Persons detected: {len(detections)}"
            cv2.putText(
                annotated, count_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

            cv2.imshow("Intrusion Detection - press 'q' to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
