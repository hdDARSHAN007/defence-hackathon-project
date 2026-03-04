import os
import time
from flask import Flask, Response
import cv2

app = Flask(__name__)


def generate_mjpeg():
    """Generator that yields MJPEG-formatted frames from logs/latest.jpg."""
    last_mtime = 0
    while True:
        latest_path = os.path.join("logs", "latest.jpg")
        if os.path.exists(latest_path):
            try:
                # check if file updated
                mtime = os.path.getmtime(latest_path)
                if mtime > last_mtime:
                    last_mtime = mtime
                    with open(latest_path, "rb") as f:
                        frame_data = f.read()
                    # yield MJPEG boundary + frame
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-length: "
                        + str(len(frame_data)).encode()
                        + b"\r\n\r\n"
                        + frame_data
                        + b"\r\n"
                    )
            except Exception:
                pass
        time.sleep(0.05)  # ~20 fps polling


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return """
    <html>
    <head><title>Defence Surveillance - Live Stream</title></head>
    <body style="font-family: Arial; text-align: center;">
        <h1>Defence Surveillance System</h1>
        <p>Live Camera Feed (MJPEG stream)</p>
        <img src="/video_feed" width="80%" style="border: 1px solid #ccc;">
        <p><small>Refresh: 50ms (20 fps)</small></p>
    </body>
    </html>
    """


if __name__ == "__main__":
    print("[Stream Server] Starting MJPEG server on http://localhost:5000")
    print("[Stream Server] View at http://localhost:5000/video_feed or http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
