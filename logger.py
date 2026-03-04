import os
import datetime
import csv
import cv2


class DetectionLogger:
    """Save detection events and screenshots to `log_dir`.

    Creates `log_dir/detections.csv` and saves screenshots as JPGs.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "detections.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "person_count", "screenshot", "alerts"])

    def log(self, person_count: int, frame=None, alerts: list | None = None):
        ts = datetime.datetime.now()
        ts_str = ts.isoformat(sep=" ", timespec="seconds")
        screenshot_path = ""
        if frame is not None:
            fname = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
            screenshot_path = os.path.join(self.log_dir, fname)
            try:
                cv2.imwrite(screenshot_path, frame)
            except Exception:
                screenshot_path = ""

        alerts_text = ";".join(alerts) if alerts else ""
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([ts_str, person_count, screenshot_path, alerts_text])
        except Exception:
            pass
