# VolumeKnuckle — Day 03

# VolumeKnuckle is a real-time gesture-based volume control system using hand tracking.

## Features
# - Control system volume using vertical fist position
# - Open hand → lock volume (no changes)
# - Hold hand still → freeze volume (prevents accidental changes)
# - Move fist to bottom → auto-mute
# - Smooth volume transitions (anti-jitter filtering)
# - Visual volume bar with real-time feedback
# - Volume zones (LOW / MEDIUM / HIGH)

## Tech Stack
# - OpenCV
# - MediaPipe Hands
# - NumPy
# - OS-level volume control (macOS / Windows / Linux)

## Controls
    # - Move fist up → increase volume
    # - Move fist down → decrease volume
    # - Open hand → lock control
    # - Hold still → freeze volume
    # - Move to bottom → mute
    # - q → Quit

## Hardware Concept
# - Analog-to-Digital Conversion (ADC)
# - Hand position acts like a potentiometer
# - Physical motion → digital signal → system output

## Notes
# - Show open hand first for detection, then close fist
# - Works best in stable lighting
# - Dead zones improve stability at min/max volume
# - Smoothing reduces noise from hand movement


import cv2
import mediapipe as mp
import numpy as np
import platform
import subprocess
import sys
import time

OS = platform.system()

def set_system_volume(percent):
    percent = max(0, min(100, int(percent)))
    try:
        if OS == "Darwin":
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {percent}"],
                capture_output=True, timeout=2
            )
        elif OS == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                volume.SetMasterVolumeLevelScalar(percent / 100.0, None)
            except ImportError:
                pass
        else:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"])
    except:
        pass


# ===================== GESTURE DETECTION =====================

def is_hand_open(hand_landmarks):
    tips = [8, 12, 16, 20]
    dips = [6, 10, 14, 18]

    extended = 0
    for tip, dip in zip(tips, dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
            extended += 1

    return extended >= 3


# ===================== SETUP =====================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

WRIST = 0
current_volume = 50
smoothed_volume = 50.0
SMOOTHING = 0.3

DEAD_ZONE_TOP = 0.10
DEAD_ZONE_BOTTOM = 0.90

# NEW: freeze logic
last_move_time = time.time()
freeze = False

def fist_to_volume(y):
    if y < DEAD_ZONE_TOP:
        return 100.0
    elif y > DEAD_ZONE_BOTTOM:
        return 0.0
    return np.interp(y, [DEAD_ZONE_TOP, DEAD_ZONE_BOTTOM], [100, 0])


print("VolumeKnuckle++ running... Press q to quit")

# ===================== LOOP =====================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    status = "NO HAND"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        y = hand.landmark[WRIST].y
        open_hand = is_hand_open(hand)

        # ===================== STATE LOGIC =====================

        if open_hand:
            status = "LOCKED"

        else:
            raw_volume = fist_to_volume(y)

            # AUTO-MUTE
            if y > 0.97:
                current_volume = 0
                set_system_volume(0)
                status = "MUTED"

            else:
                # movement detection
                if abs(raw_volume - smoothed_volume) > 2:
                    last_move_time = time.time()
                    freeze = False

                    smoothed_volume += SMOOTHING * (raw_volume - smoothed_volume)
                    current_volume = int(smoothed_volume)
                    set_system_volume(current_volume)
                    status = "ACTIVE"

                # HOLD → FREEZE
                elif time.time() - last_move_time > 1.0:
                    freeze = True
                    status = "FROZEN"

        # ===================== UI =====================

        bar_x = FRAME_W - 60
        bar_top = 50
        bar_bottom = FRAME_H - 50
        bar_height = bar_bottom - bar_top

        cv2.rectangle(frame, (bar_x, bar_top), (bar_x+30, bar_bottom), (50,50,50), -1)

        fill = int(bar_height * current_volume / 100)
        fill_top = bar_bottom - fill

        if current_volume < 30:
            color = (0,200,0)
        elif current_volume < 70:
            color = (0,220,220)
        else:
            color = (0,80,255)

        cv2.rectangle(frame, (bar_x, fill_top), (bar_x+30, bar_bottom), color, -1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x+30, bar_bottom), (255,255,255), 2)

        # TEXT
        cv2.putText(frame, f"Vol: {current_volume}%", (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.putText(frame, f"Status: {status}", (10,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("VolumeKnuckle++", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()