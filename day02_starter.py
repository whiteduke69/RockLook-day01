# AirCanvas — Day 2

# AirCanvas is a real-time drawing app using hand tracking.

## Features
# - Draw using pinch gesture (thumb + index finger)
# - Release fingers to stop drawing
# - Multiple colors (press 1, 2, 3)
# - Clear canvas with 'c'
# - Quit with 'q'

## Tech Stack
# - OpenCV
# - MediaPipe Hands
# - NumPy

## Controls
    # - Pinch → Draw
    # - Release → Stop
    # - 1 / 2 / 3 → Change color
    # - c → Clear canvas
    # - q → Quit

## Notes
# - Works best in good lighting
# - Adjust pinch threshold if needed



import cv2
import mediapipe as mp
import numpy as np
import math

# =========================
# SETUP
# =========================
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas (persistent drawing)
canvas = None

# Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8

# Drawing state
prev_x, prev_y = None, None

# Adjustable pinch threshold
PINCH_THRESHOLD = 50

# Colors (BGR)
COLORS = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
]
COLOR_NAMES = ["Green", "Red", "Blue"]
current_color = 0


def get_distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1)


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    drawing = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        landmarks = hand.landmark

        thumb = landmarks[THUMB_TIP]
        index = landmarks[INDEX_TIP]

        h, w, _ = frame.shape

        distance = get_distance(thumb, index, w, h)

        ix, iy = int(index.x * w), int(index.y * h)

        cv2.circle(frame, (ix, iy), 8, COLORS[current_color], -1)

        if distance < PINCH_THRESHOLD:
            drawing = True

            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                         COLORS[current_color], 5)

            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = None, None

        # Debug info
        cv2.putText(frame, f"Distance: {int(distance)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    else:
        prev_x, prev_y = None, None

    # Merge canvas
    frame = cv2.add(frame, canvas)

    # UI text
    status = "DRAWING" if drawing else "NOT DRAWING"
    cv2.putText(frame, status, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if drawing else (100, 100, 100), 3)

    cv2.putText(frame, f"Color: {COLOR_NAMES[current_color]}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                COLORS[current_color], 2)

    cv2.imshow("AirCanvas", frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('1'):
        current_color = 0
    elif key == ord('2'):
        current_color = 1
    elif key == ord('3'):
        current_color = 2

# Cleanup
cap.release()
cv2.destroyAllWindows()