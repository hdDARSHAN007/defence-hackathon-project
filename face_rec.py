"""
Face Recognition Module (OpenCV LBPH — no dlib needed)
=======================================================
Compares detected faces against authorized personnel using
OpenCV's LBPH (Local Binary Patterns Histograms) recognizer.

Setup:
    1. Put photos of authorized people in Authorized_persons/ (parent dir)
    2. Name files as: PersonName1.jpg, PersonName2.jpg etc.
    3. Multiple photos per person supported (name extracted before digits)
"""

import os
import re
import cv2
import numpy as np

COLOR_AUTHORIZED = (0, 255, 0)
COLOR_UNAUTHORIZED = (0, 0, 255)


class FaceRecognizer:
    def __init__(self, authorized_dir=None, confidence_threshold=80):
        # Look for Authorized_persons in parent dir by default
        if authorized_dir is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(project_dir)
            authorized_dir = os.path.join(parent_dir, "Authorized_persons")
        self.authorized_dir = authorized_dir
        self.confidence_threshold = confidence_threshold
        self.known_names = []
        self._label_map = {}
        self._name_to_label = {}

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False

        self._load_authorized_faces()

    def _extract_name(self, filename):
        """Extract person name from filename like ayush1.jpeg -> Ayush"""
        base = os.path.splitext(filename)[0]
        # Remove trailing digits
        name = re.sub(r'\d+$', '', base)
        name = name.replace("_", " ").strip().title()
        return name

    def _load_authorized_faces(self):
        if not os.path.exists(self.authorized_dir):
            print(f"[FaceRec] Directory not found: {self.authorized_dir}")
            print(f"[FaceRec] Create folder and add authorized face photos")
            return

        faces = []
        labels = []
        next_label = 0

        for fname in sorted(os.listdir(self.authorized_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(self.authorized_dir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print(f"[FaceRec] Could not read {fname}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detected = self.face_cascade.detectMultiScale(
                    gray, 1.1, 4, minSize=(30, 30)
                )

                name = self._extract_name(fname)

                # Assign label per person name
                if name not in self._name_to_label:
                    self._name_to_label[name] = next_label
                    self._label_map[next_label] = name
                    self.known_names.append(name)
                    next_label += 1

                label = self._name_to_label[name]

                if len(detected) == 0:
                    # Use whole image as face
                    face_gray = cv2.resize(gray, (200, 200))
                    faces.append(face_gray)
                    labels.append(label)
                else:
                    # Use all detected faces (in case of multiple crops)
                    for (x, y, w, h) in detected:
                        face_gray = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                        faces.append(face_gray)
                        labels.append(label)

                print(f"[FaceRec] Loaded: {fname} -> {name}")

            except Exception as e:
                print(f"[FaceRec] Error loading {fname}: {e}")

        if faces:
            self.recognizer.train(faces, np.array(labels))
            self._trained = True
            print(f"[FaceRec] Trained with {len(faces)} face samples from {len(self.known_names)} people: {', '.join(self.known_names)}")
        else:
            print(f"[FaceRec] No faces loaded from {self.authorized_dir}")

    def recognize(self, frame):
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(40, 40)
        )

        for (x, y, w, h) in detected_faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

            name = "UNAUTHORIZED"
            authorized = False
            confidence = 0.0

            if self._trained:
                label, dist = self.recognizer.predict(face_roi)
                print(f"[FaceRec DEBUG] label={self._label_map.get(label, '?')}, distance={dist:.1f}, threshold={self.confidence_threshold}")
                # LBPH: lower distance = better match
                confidence = max(0, min(1, 1.0 - dist / 200.0))

                if dist < self.confidence_threshold:
                    name = self._label_map.get(label, "UNKNOWN")
                    authorized = True

            results.append({
                "face_bbox": (x, y, x + w, y + h),
                "name": name,
                "authorized": authorized,
                "confidence": round(confidence, 2),
            })

        return results

    def draw_face_results(self, frame, face_results):
        for res in face_results:
            x1, y1, x2, y2 = res["face_bbox"]
            name = res["name"]
            authorized = res["authorized"]
            conf = res["confidence"]

            color = COLOR_AUTHORIZED if authorized else COLOR_UNAUTHORIZED
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if authorized:
                label = f"{name} ({conf:.0%})"
            else:
                label = f"UNAUTHORIZED ({conf:.0%})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 10), color, -1)
            cv2.putText(frame, label, (x1 + 3, y2 + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    @property
    def num_authorized(self):
        return len(self.known_names)

    @property
    def is_available(self):
        return True
