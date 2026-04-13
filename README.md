# Real-Time Home Monitoring System with YOLO Object Detection

## Team Members

| Role | Name | Device |
|------|------|--------|
| Sender (Laptop A) | [Edit: Your Name] | Streams webcam video via Flask |
| Receiver (Laptop B) | [Edit: Partner's Name] | Runs YOLO detection on stream |

---

## Project Overview

This project implements a real-time home monitoring pipeline across two laptops on the same WiFi network.

**Laptop A** captures live webcam video and streams it over the local network using a lightweight Flask web server.

**Laptop B** connects to that stream, receives each video frame, and runs YOLOv8 object detection on it in real time. Detected objects — people, chairs, laptops, phones, and more — are labeled directly on the video feed with bounding boxes and confidence scores.

As a bonus feature, the system automatically saves a snapshot image to disk whenever a person is detected.

### Pipeline

```
[Webcam] → [Flask Stream Server] ──WiFi──► [YOLO Detection Client] → [Live Display + Snapshots]
 Laptop A                                    Laptop B
```

---

## Project Structure

```
yolo_monitor/
├── app.py              # Laptop A — Flask webcam streaming server
├── yolo_stream.py      # Laptop B — YOLO detection client
├── README.md           # This file
└── screenshots/        # Add your demo screenshots here
    ├── laptop_a_server_running.png
    ├── laptop_b_detection_window.png
    └── snapshot_example.jpg
```

---

## How It Works

### Laptop A — Flask Streaming Server (`app.py`)

- Opens the laptop's webcam using OpenCV
- Reads frames continuously and encodes each one as a JPEG image
- Serves those frames over HTTP in MJPEG format (Motion JPEG)
- Any device on the same network can access the stream at `http://<LAPTOP_A_IP>:5000/video_feed`

### Laptop B — YOLO Detection Client (`yolo_stream.py`)

- Connects to the stream URL on Laptop A
- Reads frames from the MJPEG stream byte by byte
- Passes each frame into a YOLOv8 nano model (`yolov8n.pt`)
- Draws colored bounding boxes and confidence labels on detected objects
- Displays the annotated video in a live OpenCV window
- Prints detected object names to the terminal
- **Bonus:** Saves a timestamped JPEG snapshot whenever a person is detected

---

## Dependencies

### Laptop A (Sender)

```bash
pip install flask opencv-python
```

### Laptop B (Receiver)

```bash
pip install ultralytics opencv-python
```

> **Note:** `ultralytics` automatically downloads the YOLOv8 model file (`yolov8n.pt`, ~6MB) on first run. Internet access is required once.

---

## Setup & How to Run

### Step 1 — Find Laptop A's IP Address

On **Laptop A**, open Command Prompt and run:

```
ipconfig
```

Look for **IPv4 Address** under your active WiFi adapter. It will look something like:

```
IPv4 Address. . . . . . . . . . . : 192.168.1.42
```

Write this down. You will need it in Step 3.

---

### Step 2 — Start the Stream Server on Laptop A

Navigate to the project folder and run:

```bash
python app.py
```

Expected output:

```
=======================================================
  Home Monitoring Stream Server — Laptop A
=======================================================
  Camera Index : 0
  Resolution   : 640x480
  Port         : 5000
=======================================================
  Find your IP: run  ipconfig  (Windows)
=======================================================

 * Running on http://0.0.0.0:5000
```

You can verify the stream is working by opening a browser on Laptop A and going to:

```
http://localhost:5000/video_feed
```

---

### Step 3 — Configure Laptop B

Open `yolo_stream.py` on **Laptop B** and find this line near the top:

```python
STREAM_URL = "http://LAPTOP_A_IP:5000/video_feed"
```

Replace `LAPTOP_A_IP` with the actual IP address from Step 1. Example:

```python
STREAM_URL = "http://192.168.1.42:5000/video_feed"
```

Save the file.

---

### Step 4 — Start Detection on Laptop B

```bash
python yolo_stream.py
```

Expected output:

```
=======================================================
  YOLO Home Monitoring System — Laptop B
=======================================================
[OK] Snapshots will be saved to: ./snapshots/
[INFO] Loading YOLO model: yolov8n.pt
[OK] Model loaded. Tracking 80 object classes.

[INFO] Connecting to stream: http://192.168.1.42:5000/video_feed
[OK] Connected to stream successfully.

Starting detection. Press Q in the video window to quit.

[DETECT] person
[SNAPSHOT] Person detected — saved: snapshots/person_20250413_141523.jpg
[DETECT] person, laptop
```

A video window will open showing the live feed with bounding boxes.

---

## What the System Detects

YOLOv8 is trained on the COCO dataset and can detect 80 object classes, including:

| Category | Examples |
|----------|---------|
| People | person |
| Electronics | laptop, cell phone, TV, keyboard, mouse |
| Furniture | chair, couch, bed, dining table |
| Vehicles | car, bicycle, motorbike |
| Kitchen | bottle, cup, bowl, fork, knife |
| Animals | cat, dog, bird |

Detection boxes appear in **green** for people and **orange** for all other objects.

---

## Bonus Feature — Automatic Person Snapshot

Whenever a person is detected in the frame, the system:

1. Saves a full-resolution JPEG of that frame to the `snapshots/` folder
2. Names the file with a timestamp: `person_YYYYMMDD_HHMMSS.jpg`
3. Enforces a 5-second cooldown between saves to prevent duplicate snapshots

Snapshots folder location:

```
yolo_monitor/
└── snapshots/
    ├── person_20250413_141523.jpg
    ├── person_20250413_141531.jpg
    └── ...
```

To disable this feature, open `yolo_stream.py` and set:

```python
SAVE_PERSON_SNAPSHOTS = False
```

---

## Problems & Fixes

### Stream not opening on Laptop B

**Symptom:** `[ERROR] Could not connect to stream.`

**Fix:**
- Confirm `app.py` is running on Laptop A and shows no errors
- Double-check the IP address in `STREAM_URL`
- Make sure both laptops are on the **same WiFi network**
- On Windows, allow Python through Windows Defender Firewall:
  - Control Panel → Windows Defender Firewall → Allow an app → Add Python

---

### Camera not found on Laptop A

**Symptom:** `[ERROR] Could not open camera at index 0.`

**Fix:**
- If an external webcam is connected, try changing `CAMERA_INDEX = 1` or `CAMERA_INDEX = 2` in `app.py`
- Make sure no other application (Zoom, Teams, etc.) is currently using the camera

---

### YOLO model not downloading

**Symptom:** Error loading `yolov8n.pt`

**Fix:**
- Ensure Laptop B has an active internet connection for the first run
- The model file (~6MB) is downloaded automatically by `ultralytics` to a local cache
- After the first download it works offline

---

### Detection is very slow

**Fix:**
- `yolov8n.pt` (nano) is the fastest model. It is already selected by default.
- If still slow, reduce stream resolution in `app.py`: set `FRAME_WIDTH = 320` and `FRAME_HEIGHT = 240`
- Close other applications to free up CPU/RAM on Laptop B

---

### YOLO detects nothing

**Possible causes and fixes:**
- Confidence threshold is too high: lower `CONFIDENCE_THRESHOLD` from `0.40` to `0.25` in `yolo_stream.py`
- Poor lighting in the room — improve lighting conditions
- Objects may be too small or partially out of frame

---

## Conclusion

This project demonstrates a functional two-device IoT monitoring pipeline. Laptop A acts as an edge device — capturing and streaming raw video data — while Laptop B acts as the processing server, running AI inference in real time on the incoming stream.

The system successfully:
- Streams live webcam video over a local WiFi network using Flask
- Detects and labels multiple object categories using YOLOv8
- Annotates the video feed with bounding boxes and confidence scores
- Saves snapshot images automatically when a person is detected
- Handles errors gracefully (connection loss, camera failure)

This architecture reflects real-world patterns used in smart surveillance cameras, industrial monitoring systems, and edge AI deployments.

---

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Stream format | MJPEG over HTTP |
| Stream resolution | 640 × 480 px |
| YOLO model | YOLOv8 Nano (`yolov8n.pt`) |
| Detection classes | 80 (COCO dataset) |
| Confidence threshold | 40% |
| Snapshot cooldown | 5 seconds |
| Network port | 5000 (TCP) |
