"""
Test script for the VideoCapture module.
Run this to verify the webcam capture and preprocessing work
before integrating with the YOLO detection module.
"""

import cv2
from video_capture import VideoCapture

cam = VideoCapture(source=0, width=640, height=480)
cam.start()

try:
    for frame in cam.frames():
        cv2.imshow("Test Camera Feed - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    cam.stop()
