"""
app.py - Flask Video Streaming Server
======================================
Run this on Laptop A (the sender).
This opens your webcam and streams it over the network using Flask.
Any device on the same WiFi can view the stream in a browser or receive it with OpenCV.

Usage:
    python app.py

Stream URL:
    http://<LAPTOP_A_IP>:5000/video_feed
"""

from flask import Flask, Response
import cv2
import sys

# ─── Configuration ────────────────────────────────────────────────────────────

CAMERA_INDEX = 0          # 0 = default webcam. Try 1 or 2 if this doesn't work.
FRAME_WIDTH  = 640        # Stream resolution width
FRAME_HEIGHT = 480        # Stream resolution height
STREAM_PORT  = 5000       # Port Flask listens on
JPEG_QUALITY = 80         # JPEG compression quality (1-100). Lower = faster, less quality.

# ─── Flask App Setup ──────────────────────────────────────────────────────────

app = Flask(__name__)

# ─── Camera Initialization ────────────────────────────────────────────────────

def initialize_camera():
    """Open the webcam and configure resolution. Exits if camera is unavailable."""
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
        print("        Try changing CAMERA_INDEX to 1 or 2 in app.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print(f"[OK] Camera opened successfully at index {CAMERA_INDEX}.")
    return cap


camera = initialize_camera()

# ─── Frame Generator ──────────────────────────────────────────────────────────

def generate_frames():
    """
    Continuously read frames from the webcam and yield them as
    a multipart JPEG stream (MJPEG format).
    This is what browsers and OpenCV clients receive.
    """
    while True:
        success, frame = camera.read()

        if not success:
            print("[WARNING] Failed to read frame from camera. Retrying...")
            continue

        # Encode frame as JPEG bytes
        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not ret:
            print("[WARNING] Failed to encode frame. Skipping.")
            continue

        # Yield the frame in MJPEG multipart format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Simple home page — confirms the server is running."""
    return (
        "<h2>📷 Home Monitoring Stream Server</h2>"
        "<p>Stream is live at: <a href='/video_feed'>/video_feed</a></p>"
        "<p>Open <code>yolo_stream.py</code> on Laptop B to start detection.</p>"
    )


@app.route("/video_feed")
def video_feed():
    """
    This is the actual video stream endpoint.
    Laptop B connects to this URL to receive frames for YOLO detection.
    """
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Home Monitoring Stream Server — Laptop A")
    print("="*55)
    print(f"  Camera Index : {CAMERA_INDEX}")
    print(f"  Resolution   : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Port         : {STREAM_PORT}")
    print("="*55)
    print("  Find your IP: run  ipconfig  (Windows)")
    print("               or   hostname -I  (Linux/Mac)")
    print("  Then connect from Laptop B using that IP.")
    print("="*55 + "\n")

    # host="0.0.0.0" makes the server visible to other devices on the network
    app.run(host="0.0.0.0", port=STREAM_PORT, debug=False)
