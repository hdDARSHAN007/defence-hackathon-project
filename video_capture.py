"""
Video Capture & Frame Processing Module
========================================
Part of: AI-Based Real-Time Intrusion Detection System
Role   : Captures live video, preprocesses frames, and supplies them
         to the AI detection module (YOLOv8).

Usage (by other modules):
    from video_capture import VideoCapture

    cam = VideoCapture(source=0, width=640, height=480)
    cam.start()

    for frame in cam.frames():
        # 'frame' is a preprocessed numpy array ready for YOLO
        results = yolo_model.predict(frame)
        ...

    cam.stop()
"""

import cv2
import numpy as np
import threading
import time


class VideoCapture:
    """Threaded video capture with built-in frame preprocessing."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        brightness: float = 1.0,
        contrast: float = 1.0,
    ):
        """
        Parameters
        ----------
        source : int or str
            Camera index (0 = default webcam) or path to a video file.
        width, height : int
            Target resolution every frame is resized to before being
            handed to the detection module.
        brightness : float
            Brightness multiplier (1.0 = no change, >1 brighter, <1 darker).
        contrast : float
            Contrast multiplier  (1.0 = no change).
        """
        self.source = source
        self.width = width
        self.height = height
        self.brightness = brightness
        self.contrast = contrast

        self._cap: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()          # protects _latest_frame
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def start(self) -> "VideoCapture":
        """Open the camera and begin capturing in a background thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Error: Could not open video source '{self.source}'"
            )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[VideoCapture] Started on source={self.source}")
        return self  # allows: cam = VideoCapture().start()

    def stop(self) -> None:
        """Signal the capture thread to stop and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()
        print("[VideoCapture] Stopped and resources released")

    # ------------------------------------------------------------------
    # Internal capture loop (runs in its own thread)
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while self._running:
            if self._cap is None:
                break
            ret, frame = self._cap.read()
            if not ret:
                print("[VideoCapture] Warning: Failed to read frame, retrying…")
                time.sleep(0.1)
                continue

            processed = self.preprocess(frame)

            with self._lock:
                self._latest_frame = processed

    # ------------------------------------------------------------------
    # Frame preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply standard preprocessing to a raw camera frame:
          1. Resize to (self.width × self.height)
          2. Adjust brightness and contrast

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from OpenCV.

        Returns
        -------
        np.ndarray
            Preprocessed BGR frame ready for the detection module.
        """
        # 1. Resize
        frame = cv2.resize(frame, (self.width, self.height))

        # 2. Brightness & contrast:  output = contrast * frame + brightness_shift
        #    Using convertScaleAbs to keep values in [0, 255]
        if self.brightness != 1.0 or self.contrast != 1.0:
            brightness_shift = int((self.brightness - 1.0) * 127)
            frame = cv2.convertScaleAbs(
                frame, alpha=self.contrast, beta=brightness_shift
            )

        return frame

    # ------------------------------------------------------------------
    # Public API — getting frames
    # ------------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Return the most recent preprocessed frame (or None)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def frames(self):
        """
        Generator that yields preprocessed frames continuously.

        This is the primary interface for the detection module:

            for frame in cam.frames():
                detections = model.predict(frame)

        Yields
        ------
        np.ndarray
            The latest preprocessed frame (skips duplicates).
        """
        prev_id = -1
        while self._running:
            with self._lock:
                if self._latest_frame is None:
                    continue
                frame = self._latest_frame.copy()
                cur_id = id(self._latest_frame)

            if cur_id == prev_id:
                # same frame object — wait briefly to avoid busy‑spinning
                time.sleep(0.005)
                continue

            prev_id = cur_id
            yield frame

    # ------------------------------------------------------------------
    # Context‑manager support (optional convenience)
    # ------------------------------------------------------------------
    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


# ======================================================================
# Stand‑alone test — run this file directly to verify the webcam works
# ======================================================================
if __name__ == "__main__":
    cam = VideoCapture(source=0, width=640, height=480, brightness=1.0)
    cam.start()

    try:
        for frame in cam.frames():
            cv2.imshow("Live Camera Feed — press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()