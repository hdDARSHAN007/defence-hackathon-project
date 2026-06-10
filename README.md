# Defence Hackathon Project

AI-based real-time surveillance system for border/base monitoring using:
- **YOLOv8** for person and suspicious-object detection
- **OpenCV LBPH** face recognition for authorized vs unauthorized detection
- **Rule engine + alerting** for intruder, loitering, restricted-area, and night-movement events
- **Dashboard + stream server** for live monitoring

## Project purpose
This project processes live camera frames, detects people/threat objects, checks faces against an authorized-person dataset, and raises alerts for suspicious situations in near real time.

## Repository structure
- `main.py` - core runtime pipeline
- `video_capture.py` - threaded camera capture and preprocessing
- `detection.py` - YOLO-based person/object detection + drawing
- `face_rec.py` - authorized face loading and recognition (LBPH)
- `rules.py` - alert/risk rules
- `alerts.py` - console/sound/email/telegram/dashboard alerts
- `dashboard.py` - Streamlit command-center dashboard
- `stream_server.py` - Flask MJPEG stream from `logs/latest.jpg`
- `logger.py` - CSV + screenshot logging
- `evaluate_metrics.py` - offline evaluation metrics (accuracy/precision/recall/F1/confusion matrix)

## Dependencies
The codebase currently uses these Python packages:
- `opencv-contrib-python` (for `cv2.face.LBPHFaceRecognizer_create`)
- `ultralytics`
- `numpy`
- `pandas`
- `streamlit`
- `flask`
- `requests`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install opencv-contrib-python ultralytics numpy pandas streamlit flask requests
```

Download/keep model file at repository root:
- `yolov8n.pt`

Add authorized face images in a sibling directory (as expected by `face_rec.py`):
```text
<parent_of_repo>/Authorized_persons/
```
Example names: `Ayush1.jpg`, `Ayush2.jpg`, `Guard1.jpg`.

## Run the system
### 1) Start detection pipeline
```bash
python main.py
```

### 2) Optional: start dashboard
```bash
streamlit run dashboard.py
```

### 3) Optional: start MJPEG stream
```bash
python stream_server.py
```
Open `http://localhost:5000`.

## Usage notes
- Alerts and state are written under `logs/` (`detections.csv`, `state.json`, `alert_history.json`, screenshots).
- Press `q` in OpenCV windows to quit.

## Model evaluation metrics
The real-time runtime (`main.py`) focuses on live detection/alerts and **does not compute offline classification metrics from labeled ground-truth data by default**.

For dataset-based evaluation, use `evaluate_metrics.py`.

### Input format
Prepare a CSV file with at least:
- `y_true` (ground truth label)
- `y_pred` (model prediction)

Binary labels are expected (e.g., `AUTHORIZED` / `UNAUTHORIZED`, or `1` / `0`).

### Command
```bash
python evaluate_metrics.py \
  --input logs/evaluation_labels.csv \
  --true-col y_true \
  --pred-col y_pred \
  --positive-label UNAUTHORIZED
```

### Metrics computed
- **Accuracy** = `(TP + TN) / (TP + TN + FP + FN)`
- **Precision** = `TP / (TP + FP)`
- **Recall** = `TP / (TP + FN)`
- **F1 score** = `2 * (Precision * Recall) / (Precision + Recall)`
- **Confusion matrix** in the format `[[TN, FP], [FN, TP]]`

### Sample output
```text
Total samples: 20
Positive label: UNAUTHORIZED

Accuracy : 0.9000 (90.00%)
Precision: 0.8750
Recall   : 0.8750
F1 score : 0.8750

Confusion Matrix [[TN, FP], [FN, TP]]
[[11, 1], [1, 7]]
```

This evaluation step can be run after collecting predictions from experiments or replay pipelines, and gives a standard quality snapshot for model behavior.
