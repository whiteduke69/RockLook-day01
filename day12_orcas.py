# SnapAnnotator ELITE — Day 12
# SnapAnnotator is a real-time vision AI system that captures webcam frames, detects objects using YOLO, and reasons about them using a local vision-language model.
# Features
# Real-time webcam processing
# Object detection with bounding boxes (YOLOv8)
# Local vision-language reasoning (moondream via Ollama)
# Interactive object querying (press 1–9)
# Smooth UI overlay with labels and boxes
# Confidence scores for detected objects
# Latency tracking (inference time display)
# Hybrid perception + reasoning pipeline
# Tech Stack
# Python
# OpenCV (camera + rendering)
# Ollama (local LLM runtime)
# YOLOv8 (Ultralytics object detection model)
# subprocess (model calls)
# regex (structured parsing)
# Controls
# SPACE → Capture frame + analyze scene
# 1–9 → Ask about detected object
# q → Quit
# Hardware Concept
# Edge vision pipeline (camera → inference → action)
# Detection + reasoning split:
# YOLO → spatial understanding (where objects are)
# VLM → semantic understanding (what + meaning)
# Frame resizing simulates bandwidth constraints
# Latency measurement reflects real-time system limits
# Mirrors embedded AI systems (Jetson Nano, edge cameras)


# Pipeline
# Camera → Frame → Resize → YOLO → Bounding Boxes
# ↓
# Moondream (VLM)
# ↓
# Reasoning + Interaction
# ↓
# Output
# Notes
# Requires ollama serve running in background
# Model must be pulled: ollama pull moondream
# YOLO model auto-downloads (yolov8n.pt)
# Inference speed depends on CPU (~1–5 FPS typical)
# Resizing to 512px improves performance significantly
# Combines deterministic detection + generative reasoning
# Why This Project Is Strong
# Multi-model system (not just CV)
# Interactive AI agent (not static detection)
# Simulates real hardware pipeline
# Combines perception, reasoning, and interaction
# Positioning
# SnapAnnotator represents a mini edge AI system similar to smart cameras, autonomous perception modules, and AR object understanding systems.



import cv2
import subprocess
import tempfile
import os
import sys
import time
import re
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

MODEL = "moondream"
MAX_IMAGE_SIZE = 512

DESCRIBE_PROMPT = (
    "Analyze this image and return a numbered list of visible objects.\n"
    "Rules:\n"
    "- Max 5 objects\n"
    "- Each line must be: number. object_name\n"
    "- No extra text\n"
)

# Load YOLO model (lightweight)
yolo_model = YOLO("yolov8n.pt")

# ============================================================
# SETUP CHECK
# ============================================================

def check_setup():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("❌ Run: ollama serve")
            sys.exit(1)
        if "moondream" not in result.stdout.lower():
            print("❌ Run: ollama pull moondream")
            sys.exit(1)
        print("✅ System ready")
    except:
        print("❌ Ollama not installed")
        sys.exit(1)

# ============================================================
# VLM QUERY
# ============================================================

def query_vlm(image_path, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, f"{prompt} {image_path}"],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except:
        return "[Error]"

# ============================================================
# IMAGE PROCESSING
# ============================================================

def resize_and_save(frame):
    h, w = frame.shape[:2]
    scale = MAX_IMAGE_SIZE / max(h, w)

    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp.name, frame)
    return temp.name

# ============================================================
# YOLO DETECTION
# ============================================================

def detect_objects(frame):
    results = yolo_model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = yolo_model.names[cls]

        detections.append({
            "label": label,
            "conf": conf,
            "box": (x1, y1, x2, y2)
        })

    return detections

# ============================================================
# DRAW BOXES
# ============================================================

def draw_boxes(frame, detections):
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["conf"]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Label text
        text = f"{i+1}. {label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)

    return frame

# ============================================================
# PARSER
# ============================================================

def parse_object_list(text):
    objects = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(\d+[\.\)]\s*)(.+)", line)
        if match:
            objects.append(match.group(2).strip())

    return objects[:9]

# ============================================================
# MAIN
# ============================================================

def main():
    check_setup()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ No webcam")
        return

    print("\n📸 SnapAnnotator ELITE Started")
    print("SPACE = analyze | 1-9 = ask | q = quit\n")

    last_detections = []
    latency = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detect_objects(frame)
        display = draw_boxes(frame.copy(), detections)

        # Latency display
        if latency > 0:
            cv2.putText(display, f"Inference: {latency:.1f}s",
                        (10, display.shape[0]-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,0), 1)

        # Controls
        cv2.putText(display, "SPACE=analyze  1-9=ask  q=quit",
                    (10, display.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200,200,200), 1)

        cv2.imshow("SnapAnnotator ELITE", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Analyze scene with VLM
        elif key == ord(' '):
            print("\n📸 Capturing + analyzing...")

            img_path = resize_and_save(frame)

            start = time.time()
            response = query_vlm(img_path, DESCRIBE_PROMPT)
            latency = time.time() - start

            os.unlink(img_path)

            print(f"\n⚡ {latency:.1f}s")
            print(response)

            last_detections = detections

        # Ask about detected object
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')

            if idx < len(last_detections):
                obj = last_detections[idx]["label"]
                print(f"\n🔍 Asking about: {obj}")

                img_path = resize_and_save(frame)

                followup_prompt = f"""
About the {obj} in this image:
- What is it used for?
- Where is it located?
- One interesting fact.
Keep it short.
"""

                start = time.time()
                answer = query_vlm(img_path, followup_prompt)
                latency = time.time() - start

                os.unlink(img_path)

                print(f"\n⚡ {latency:.1f}s")
                print(answer)
            else:
                print("❌ No such object")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Closed")

# ============================================================

if __name__ == "__main__":
    main()