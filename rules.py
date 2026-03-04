import time
import math
import cv2
import numpy as np


class RuleEngine:
    """Simple rules engine for person-based alerts.

    Rules implemented:
      - person_count > 1 -> suspicious activity
      - person in restricted area -> restricted-area alert
      - person stays > `stay_seconds` -> loitering alert
      - movement detected at night -> night-movement alert
    """

    def __init__(self, stay_seconds: float = 10.0, night_brightness: float = 40.0):
        self.stay_seconds = stay_seconds
        self.night_brightness = night_brightness

        self._tracks = {}  # id -> {centroid, first_seen, last_seen, bboxes}
        self._next_id = 1
        self._prev_gray = None
        self.initialized = False
        self.restricted_area = None  # (x1,y1,x2,y2)

    # ----------------------- helpers ---------------------------------
    @staticmethod
    def _centroid_from_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ----------------------- public API -------------------------------
    def update(self, detections, frame, timestamp=None):
        """Update engine state with the latest detections and frame.

        Returns a list of alert strings (may be empty).
        """
        if timestamp is None:
            timestamp = time.time()

        h, w = frame.shape[:2]
        if not self.initialized:
            # default restricted area: central rectangle (configurable later)
            self.restricted_area = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
            self.initialized = True

        alerts = []

        # Rule: person count > 1
        if len(detections) > 1:
            alerts.append("Suspicious activity: multiple persons detected")

        # Associate detections with simple centroid tracker
        curr_centroids = []
        for det in detections:
            c = self._centroid_from_bbox(det["bbox"])
            curr_centroids.append((det, c))

        matched_ids = set()
        # greedy nearest assignment
        for det, centroid in curr_centroids:
            best_id = None
            best_dist = 1e9
            for tid, t in self._tracks.items():
                d = self._dist(centroid, t["centroid"])
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None and best_dist < 50:  # pixels threshold
                # update track
                t = self._tracks[best_id]
                t["centroid"] = centroid
                t["last_seen"] = timestamp
                t["bboxes"].append(det["bbox"])
                matched_ids.add(best_id)
                det["track_id"] = best_id
            else:
                # new track
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "centroid": centroid,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "bboxes": [det["bbox"]],
                }
                det["track_id"] = tid
                matched_ids.add(tid)

        # Remove stale tracks (not seen for > 3s)
        stale = []
        for tid, t in list(self._tracks.items()):
            if tid not in matched_ids and (timestamp - t["last_seen"]) > 3.0:
                stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        # Rule: person in restricted area
        rx1, ry1, rx2, ry2 = self.restricted_area
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = self._centroid_from_bbox(det["bbox"])
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                alerts.append(f"Restricted area breach: track {det.get('track_id')} at ({cx},{cy})")

        # Rule: person stays > stay_seconds (loitering)
        for tid, t in self._tracks.items():
            duration = timestamp - t["first_seen"]
            if duration >= self.stay_seconds:
                # small movement check: compare recent bbox centers
                if len(t["bboxes"]) >= 2:
                    c1 = self._centroid_from_bbox(t["bboxes"][-1])
                    c0 = self._centroid_from_bbox(t["bboxes"][0])
                    if self._dist(c0, c1) < 30:  # stationary threshold
                        alerts.append(f"Loitering: track {tid} present {int(duration)}s")

        # Rule: movement detected at night
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        movement_alert = False
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, th = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_pixels = int(np.count_nonzero(th))
            motion_fraction = motion_pixels / (w * h)
            if mean_brightness < self.night_brightness and motion_fraction > 0.002:
                movement_alert = True
        self._prev_gray = gray

        if movement_alert:
            alerts.append("Night movement detected")

        # deduplicate alerts
        unique = []
        seen = set()
        for a in alerts:
            if a not in seen:
                unique.append(a)
                seen.add(a)

        return unique
