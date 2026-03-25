import cv2
import mediapipe as mp
import pygame
import os
import sys

# =========================
# CONFIG (YOU CAN ADJUST)
# =========================
GAZE_THRESHOLD = 0.015
MUSIC_FILE = "music.mp3"

# =========================
# INIT
# =========================

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Webcam not found")
    sys.exit(1)

# FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# Audio
pygame.mixer.init()
music_loaded = False

if os.path.exists(MUSIC_FILE):
    pygame.mixer.music.load(MUSIC_FILE)
    music_loaded = True
    print(f"🎵 Loaded: {MUSIC_FILE}")
else:
    print("⚠️ music.mp3 not found (no sound will play)")

# Landmarks
LEFT_IRIS = 468
RIGHT_IRIS = 473
NOSE_TIP = 1

is_playing = False

print("\n🎸 RockLook Running")
print(f"Threshold = {GAZE_THRESHOLD}")
print("Look DOWN → Play | Look UP → Pause")
print("Press 'q' to quit\n")

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        iris_y = (landmarks[LEFT_IRIS].y + landmarks[RIGHT_IRIS].y) / 2
        nose_y = landmarks[NOSE_TIP].y
        gaze_offset = iris_y - nose_y

        looking_down = gaze_offset < -GAZE_THRESHOLD

        # =========================
        # SENSOR → THRESHOLD → ACTUATOR
        # =========================
        if music_loaded:
            if looking_down and not is_playing:
                pygame.mixer.music.play()
                is_playing = True
                print("▶ PLAY")

            elif not looking_down and is_playing:
                pygame.mixer.music.pause()
                is_playing = False
                print("⏸ PAUSE")

        # =========================
        # DISPLAY
        # =========================
        status = "DOWN" if looking_down else "UP"
        color = (0, 0, 255) if looking_down else (0, 255, 0)

        cv2.putText(frame, f"Gaze: {gaze_offset:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Threshold: {GAZE_THRESHOLD}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Status: {status}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("RockLook", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("\n👋 Program ended")