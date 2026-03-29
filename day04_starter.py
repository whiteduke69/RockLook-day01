# BlinkLock — Day 04

# BlinkLock is a real-time vision-based locking system using facial landmark tracking and blink detection.

## Features
# - Detect eye closure using Eye Aspect Ratio (EAR)
# - 3 rapid blinks → lock system
# - Wink → unlock system
# - Face absence → auto-lock
# - Debounce logic prevents false blink triggers
# - State machine ensures stable behavior (IDLE / COUNTING / LOCKED)
# - Real-time EAR display and system status feedback

## Tech Stack
# - OpenCV
# - MediaPipe FaceMesh
# - Python

## Controls
    # - Blink 3 times rapidly → Lock
    # - Wink (one eye) → Unlock
    # - u → Manual unlock (fallback)
    # - q → Quit

## Hardware Concept
# - Digital signal processing (eye open/close detection)
# - Debounce logic (filters noisy input like physical buttons)
# - Multi-event sequencing (3 blinks within time window)
# - State machine (IDLE → COUNTING → LOCKED)

## Notes
# - Adjust EAR threshold depending on your eye shape
# - Works best in good lighting conditions
# - Avoid partial eye closures to prevent false triggers
# - Blink timing window affects responsiveness and accuracy


import cv2
import mediapipe as mp
import time
import sys

# ==============================
# SETUP
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# Eye landmarks
LEFT_TOP = [159, 160, 161]
LEFT_BOTTOM = [145, 144, 153]
LEFT_LEFT = 33
LEFT_RIGHT = 133

RIGHT_TOP = [386, 387, 388]
RIGHT_BOTTOM = [374, 373, 380]
RIGHT_LEFT = 362
RIGHT_RIGHT = 263

# ==============================
# CONFIG
# ==============================
EAR_THRESHOLD = 0.23
BLINK_WINDOW = 2.0
BLINK_TARGET = 3
MIN_FRAMES = 2
COOLDOWN = 1.0
FACE_TIMEOUT = 5.0

# ==============================
# STATE MACHINE
# ==============================
STATE_IDLE = "IDLE"
STATE_COUNTING = "COUNTING"
STATE_LOCKED = "LOCKED"

state = STATE_IDLE
blink_count = 0
start_time = 0
closed_frames = 0
last_action_time = 0
last_face_time = time.time()

# ==============================
# FUNCTIONS
# ==============================

def get_ear(landmarks, top, bottom, left, right):
    vertical = 0
    for t, b in zip(top, bottom):
        vertical += abs(landmarks[t].y - landmarks[b].y)
    vertical /= len(top)

    horizontal = abs(landmarks[left].x - landmarks[right].x)
    return vertical / horizontal if horizontal != 0 else 0


def handle_command(cmd):
    global state, blink_count, last_action_time

    now = time.time()
    if now - last_action_time < COOLDOWN:
        return

    if cmd == "LOCK":
        state = STATE_LOCKED
        blink_count = 0
        print("🔒 LOCKED")

    elif cmd == "UNLOCK":
        state = STATE_IDLE
        blink_count = 0
        print("🔓 UNLOCKED")

    last_action_time = now


# ==============================
# MAIN LOOP
# ==============================
print("BlinkLock v2.0 running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    current_ear = 0
    left_ear = 0
    right_ear = 0

    # ==========================
    # FACE DETECTION
    # ==========================
    if results.multi_face_landmarks:
        last_face_time = time.time()
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = get_ear(landmarks, LEFT_TOP, LEFT_BOTTOM, LEFT_LEFT, LEFT_RIGHT)
        right_ear = get_ear(landmarks, RIGHT_TOP, RIGHT_BOTTOM, RIGHT_LEFT, RIGHT_RIGHT)

        current_ear = (left_ear + right_ear) / 2

        eye_closed = current_ear < EAR_THRESHOLD

        # ==========================
        # BLINK DETECTION
        # ==========================
        if eye_closed:
            closed_frames += 1
        else:
            if closed_frames >= MIN_FRAMES:
                # VALID BLINK
                if state == STATE_IDLE:
                    state = STATE_COUNTING
                    blink_count = 1
                    start_time = time.time()

                elif state == STATE_COUNTING:
                    blink_count += 1

                    if blink_count >= BLINK_TARGET:
                        handle_command("LOCK")

            closed_frames = 0

        # ==========================
        # WINK DETECTION (UNLOCK)
        # ==========================
        if state == STATE_LOCKED:
            if left_ear < EAR_THRESHOLD and right_ear > EAR_THRESHOLD:
                handle_command("UNLOCK")

    # ==========================
    # FACE ABSENCE AUTO LOCK
    # ==========================
    if time.time() - last_face_time > FACE_TIMEOUT:
        handle_command("LOCK")

    # ==========================
    # TIMEOUT RESET
    # ==========================
    if state == STATE_COUNTING:
        if time.time() - start_time > BLINK_WINDOW:
            state = STATE_IDLE
            blink_count = 0

    # ==========================
    # DISPLAY
    # ==========================
    if state == STATE_LOCKED:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.putText(frame, "LOCKED", (w//2 - 120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Info panel
    cv2.putText(frame, f"EAR: {current_ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(frame, f"State: {state}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"Blinks: {blink_count}/{BLINK_TARGET}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

    cv2.putText(frame, "3 blinks = LOCK | wink = UNLOCK", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    cv2.imshow("OrcaOS - BlinkLock", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u'):
        handle_command("UNLOCK")

cap.release()
cv2.destroyAllWindows()