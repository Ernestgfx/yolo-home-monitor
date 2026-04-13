"""
yolo_stream.py - YOLO Object Detection Client
===============================================
Run this on Laptop B (the receiver).
This connects to the Flask stream on Laptop A, runs YOLOv8 detection
on every frame, displays the results in a window, and saves a snapshot
whenever a person is detected.

Usage:
    python yolo_stream.py

Before running:
    1. Make sure app.py is running on Laptop A.
    2. Set STREAM_URL below to Laptop A's IP address.

Dependencies:
    pip install ultralytics opencv-python requests
"""

import cv2
import urllib.request
import numpy as np
import os
import time
from datetime import datetime
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

# ⚠️  CHANGE THIS: Replace with the actual IP address of Laptop A.
#     Example: "http://192.168.1.42:5000/video_feed"
STREAM_URL = "http://LAPTOP_A_IP:5000/video_feed"

# YOLO model to use. "yolov8n.pt" is the smallest/fastest (nano).
# Options: yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt
YOLO_MODEL = "yolov8n.pt"

# Confidence threshold — detections below this % are ignored (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.40

# ─── Bonus Feature: Person Detection Snapshot ─────────────────────────────────

SAVE_PERSON_SNAPSHOTS = True          # Set to False to disable saving
SNAPSHOT_FOLDER      = "snapshots"    # Folder where snapshots are saved
SNAPSHOT_COOLDOWN    = 5              # Minimum seconds between snapshots (avoid spam)

# ─── Display Settings ─────────────────────────────────────────────────────────

# Bounding box and label colors (BGR format)
COLOR_PERSON  = (0,   255,  0)    # Green for person
COLOR_OBJECT  = (0,   165, 255)   # Orange for everything else
COLOR_TEXT_BG = (0,     0,   0)   # Black label background

WINDOW_TITLE = "YOLO Home Monitor — Laptop B (press Q to quit)"

# ─── Setup ────────────────────────────────────────────────────────────────────

def setup_snapshot_folder():
    """Create the snapshots folder if it doesn't exist."""
    if SAVE_PERSON_SNAPSHOTS:
        os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
        print(f"[OK] Snapshots will be saved to: ./{SNAPSHOT_FOLDER}/")


def load_model():
    """Load the YOLOv8 model. Downloads automatically on first run."""
    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    print("       (This may download the model on first run — ~6MB)")
    model = YOLO(YOLO_MODEL)
    print(f"[OK] Model loaded. Tracking {len(model.names)} object classes.")
    return model


def connect_to_stream():
    """
    Open a connection to the MJPEG stream from Laptop A.
    Returns a urllib stream object or exits if the connection fails.
    """
    print(f"\n[INFO] Connecting to stream: {STREAM_URL}")
    try:
        stream = urllib.request.urlopen(STREAM_URL, timeout=10)
        print("[OK] Connected to stream successfully.\n")
        return stream
    except Exception as e:
        print(f"\n[ERROR] Could not connect to stream.")
        print(f"        Reason : {e}")
        print(f"        URL    : {STREAM_URL}")
        print("\nTroubleshooting:")
        print("  1. Make sure app.py is running on Laptop A.")
        print("  2. Check that LAPTOP_A_IP in this file is correct.")
        print("  3. Both laptops must be on the same WiFi network.")
        print("  4. Check Windows Firewall — allow Python on port 5000.\n")
        raise SystemExit(1)

# ─── Frame Parsing ────────────────────────────────────────────────────────────

def read_mjpeg_frame(stream):
    """
    Parse a single JPEG frame from the MJPEG stream.
    MJPEG streams separate frames using boundary markers.
    Returns a decoded OpenCV frame (numpy array), or None on failure.
    """
    bytes_buffer = b""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            return None
        bytes_buffer += chunk

        # JPEG frames start with FFD8 and end with FFD9
        start = bytes_buffer.find(b"\xff\xd8")
        end   = bytes_buffer.find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:
            jpg_bytes = bytes_buffer[start:end + 2]
            bytes_buffer = bytes_buffer[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            return frame

# ─── Drawing Utilities ────────────────────────────────────────────────────────

def draw_label(frame, text, x, y, color):
    """Draw a filled label box above a bounding box."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Background rectangle
    cv2.rectangle(frame, (x, y - text_h - 6), (x + text_w + 4, y), COLOR_TEXT_BG, -1)
    # Text
    cv2.putText(frame, text, (x + 2, y - 3), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, results, model_names):
    """
    Draw bounding boxes and labels for all detections on the frame.
    Returns a list of detected class names for this frame.
    """
    detected_labels = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id   = int(box.cls[0])
            label_name = model_names[class_id]
            label_text = f"{label_name} {confidence:.0%}"

            detected_labels.append(label_name)

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Use green for person, orange for everything else
            color = COLOR_PERSON if label_name == "person" else COLOR_OBJECT

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label_text, x1, y1, color)

    return detected_labels


def draw_hud(frame, detected_labels, fps):
    """Draw a heads-up display overlay: FPS, frame timestamp, object count."""
    h, w = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top-left: FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Top-right: timestamp
    cv2.putText(frame, timestamp, (w - 210, 25),
                font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom-left: detection summary
    if detected_labels:
        summary = "Detected: " + ", ".join(set(detected_labels))
        cv2.putText(frame, summary, (10, h - 12),
                    font, 0.55, (0, 255, 180), 1, cv2.LINE_AA)

# ─── Bonus: Save Snapshot on Person Detection ─────────────────────────────────

last_snapshot_time = 0

def maybe_save_snapshot(frame, detected_labels):
    """
    If a 'person' is in the detected labels and enough time has passed
    since the last snapshot, save the frame as a JPEG image.
    """
    global last_snapshot_time

    if not SAVE_PERSON_SNAPSHOTS:
        return
    if "person" not in detected_labels:
        return

    now = time.time()
    if now - last_snapshot_time < SNAPSHOT_COOLDOWN:
        return  # Cooldown not yet elapsed

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(SNAPSHOT_FOLDER, f"person_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"[SNAPSHOT] Person detected — saved: {filename}")
    last_snapshot_time = now

# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  YOLO Home Monitoring System — Laptop B")
    print("="*55)

    setup_snapshot_folder()
    model  = load_model()
    stream = connect_to_stream()

    print("Starting detection. Press Q in the video window to quit.\n")

    fps_timer   = time.time()
    frame_count = 0
    fps         = 0.0

    while True:
        frame = read_mjpeg_frame(stream)

        if frame is None:
            print("[WARNING] Lost connection to stream. Attempting to reconnect...")
            try:
                stream = connect_to_stream()
            except SystemExit:
                break
            continue

        # ── Run YOLO detection ──────────────────────────────────────────────
        results = model(frame, verbose=False)

        # ── Draw detections on frame ────────────────────────────────────────
        detected_labels = draw_detections(frame, results, model.names)

        # ── Print detected objects to terminal ─────────────────────────────
        if detected_labels:
            unique = list(set(detected_labels))
            print(f"[DETECT] {', '.join(unique)}")

        # ── Bonus: Save snapshot if person detected ─────────────────────────
        maybe_save_snapshot(frame, detected_labels)

        # ── FPS calculation ─────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            frame_count = 0
            fps_timer   = time.time()

        # ── Draw HUD overlay ────────────────────────────────────────────────
        draw_hud(frame, detected_labels, fps)

        # ── Show frame ──────────────────────────────────────────────────────
        cv2.imshow(WINDOW_TITLE, frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit signal received. Closing...")
            break

    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")


if __name__ == "__main__":
    main()
