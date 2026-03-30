# FaceEQ+ — Day 05

# FaceEQ+ is a real-time vision-based audio control system that maps head movement to continuous input signals, turning your head into a rotary encoder.

## Features #

# - Head yaw (left/right) → scrub audio position  
# - Head pitch (up/down) → control playback speed  
# - Face distance → control volume  
# - Smooth scrubbing for analog-like control  
# - Dead zones to reduce jitter from small movements  
# - Auto pause when no face detected  
# - Real-time visual feedback (status, speed, position, volume)  
# - Keyboard fallback controls  

## Tech Stack #

# - OpenCV  
# - MediaPipe FaceMesh  
# - pygame mixer  
# - Python  

## Controls #

    # - Turn head LEFT → rewind  
    # - Turn head RIGHT → fast forward  
    # - Tilt head UP → increase speed  
    # - Tilt head DOWN → decrease speed  
    # - Move closer → increase volume  
    # - Move farther → decrease volume  

# - SPACE → play / pause  
# - r → reset track  
# - q → quit  
# - Arrow keys → manual fallback control  

## Hardware Concept #

# - Rotary encoder (head rotation → continuous value output)  
# - Multi-axis input (yaw + pitch + depth)  
# - Analog signal mapping (angle → speed/position)  
# - Dead zone filtering (like joystick stabilization)  
# - Continuous control system (real-time input stream)  

## Notes #

# - Playback speed is simulated (pygame limitation)  
# - Minor audio artifacts during scrubbing are normal  
# - Keep face centered for stable tracking  
# - Works best with consistent lighting  
# - Adjust dead zones and scaling for better responsiveness  



import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import os
import time

# ============== SETUP =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

pygame.mixer.init(frequency=44100)

TRACK_FILE = "track.mp3"
if not os.path.exists(TRACK_FILE):
    print("ERROR: track.mp3 not found")
    sys.exit(1)

pygame.mixer.music.load(TRACK_FILE)

# LANDMARKS
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FOREHEAD = 10

# ================ SETTINGS =================
YAW_DEAD_ZONE = 5.0
SCRUB_SPEED = 0.2

PITCH_DEAD_ZONE = 5.0
MIN_SPEED = 0.5
MAX_SPEED = 2.0

TRACK_LENGTH = 180.0

# ================= STATE =================
is_playing = True
track_position = 0.0
playback_speed = 1.0

smooth_yaw = 0.0
smooth_pitch = 0.0
SMOOTHING = 0.25

last_face_time = time.time()
volume = 0.5

pygame.mixer.music.play()

# ================= FUNCTIONS =================
def estimate_head_pose(landmarks):
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    eye_mid_x = (left_eye.x + right_eye.x) / 2
    eye_distance = abs(right_eye.x - left_eye.x)

    yaw = ((nose.x - eye_mid_x) / eye_distance) * 60 if eye_distance > 0 else 0

    nose_to_chin = chin.y - nose.y
    forehead_to_nose = nose.y - forehead.y

    pitch = ((nose_to_chin / forehead_to_nose) - 1.0) * 40 if forehead_to_nose > 0 else 0

    return yaw, pitch

# ================= MAIN LOOP =================
print("FaceEQ+ running...")
print("Controls: SPACE play/pause | r reset | q quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True
        last_face_time = time.time()

        landmarks = results.multi_face_landmarks[0].landmark
        raw_yaw, raw_pitch = estimate_head_pose(landmarks)

        # smoothing
        smooth_yaw += SMOOTHING * (raw_yaw - smooth_yaw)
        smooth_pitch += SMOOTHING * (raw_pitch - smooth_pitch)

        current_yaw = smooth_yaw
        current_pitch = smooth_pitch

        # ===== SCRUB (SMOOTHED) =====
        if abs(current_yaw) > YAW_DEAD_ZONE:
            scrub_amount = current_yaw - (YAW_DEAD_ZONE if current_yaw > 0 else -YAW_DEAD_ZONE)
            scrub_velocity = scrub_amount / 25.0
            track_position += scrub_velocity * SCRUB_SPEED
            track_position = np.clip(track_position, 0, TRACK_LENGTH)
            pygame.mixer.music.set_pos(track_position)

        # ===== SPEED =====
        if abs(current_pitch) > PITCH_DEAD_ZONE:
            pitch_amount = current_pitch - (PITCH_DEAD_ZONE if current_pitch > 0 else -PITCH_DEAD_ZONE)
            speed_offset = (pitch_amount / 20.0)
            playback_speed = np.clip(1.0 + speed_offset, MIN_SPEED, MAX_SPEED)
        else:
            playback_speed = 1.0

        # ===== VOLUME (FACE SIZE) =====
        face_width = abs(landmarks[RIGHT_EYE_OUTER].x - landmarks[LEFT_EYE_OUTER].x)
        volume = np.clip((face_width - 0.05) / 0.25, 0.0, 1.0)
        pygame.mixer.music.set_volume(volume)

        # ===== VISUAL NOSE =====
        nose = landmarks[NOSE_TIP]
        nx, ny = int(nose.x * FRAME_W), int(nose.y * FRAME_H)
        cv2.circle(frame, (nx, ny), 6, (0,255,255), -1)

    # ===== AUTO PAUSE =====
    if time.time() - last_face_time > 2:
        if is_playing:
            pygame.mixer.music.pause()
            is_playing = False

    # ===== UI =====
    status = "PLAYING" if is_playing else "PAUSED"
    cv2.putText(frame, status, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,255,0) if is_playing else (0,0,255), 2)

    cv2.putText(frame, f"Speed: {playback_speed:.2f}x", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

    cv2.putText(frame, f"Pos: {track_position:.1f}s", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    # ===== VOLUME BAR =====
    bar_x, bar_y = 10, 120
    bar_w, bar_h = 150, 10

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (80,80,80), 1)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * volume), bar_y + bar_h),
                  (0,255,0), -1)

    cv2.putText(frame, f"Volume: {int(volume*100)}%",
                (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    if not face_detected:
        cv2.putText(frame, "No face detected",
                    (10, FRAME_H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("FaceEQ+ (OrcaOS)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if is_playing:
            pygame.mixer.music.pause()
            is_playing = False
        else:
            pygame.mixer.music.unpause()
            is_playing = True
    elif key == ord('r'):
        track_position = 0.0
        pygame.mixer.music.rewind()

    # Arrow keys fallback
    elif key == 81:
        track_position = max(0, track_position - 2)
    elif key == 83:
        track_position = min(TRACK_LENGTH, track_position + 2)
    elif key == 82:
        playback_speed = min(MAX_SPEED, playback_speed + 0.1)
    elif key == 84:
        playback_speed = max(MIN_SPEED, playback_speed - 0.1)

    # Update position
    if is_playing:
        track_position += (1/30.0) * playback_speed
        track_position = min(track_position, TRACK_LENGTH)

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()