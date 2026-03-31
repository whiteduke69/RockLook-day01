# Day 06 — BreathClock

# Features:
# - Real-time breathing detection using microphone
# - Butterworth low-pass filtering
# - Automatic threshold calibration
# - BPM calculation with smoothing
# - Breath phase detection (inhale/exhale)
# - Session logging

# Tech:
# - PyAudio
# - SciPy Signal Processing
# - Matplotlib Animation




import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import collections
import sys

# ============================================================
# CONFIG
# ============================================================

RATE = 44100
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1

CUTOFF_HZ = 0.4
FILTER_ORDER = 2

AUTO_CALIBRATION_TIME = 5  # seconds
HISTORY_LENGTH = 500

# ============================================================
# AUDIO SETUP
# ============================================================

pa = pyaudio.PyAudio()

device_index = None
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        device_index = i
        print(f"Using mic: {info['name']}")
        break

if device_index is None:
    print("No mic found")
    sys.exit(1)

stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=CHUNK,
)

# ============================================================
# FILTER SETUP
# ============================================================

effective_rate = RATE / CHUNK
nyquist = effective_rate / 2
normalized_cutoff = min(CUTOFF_HZ / nyquist, 0.95)

b, a = butter(FILTER_ORDER, normalized_cutoff, btype='low')
zi = np.zeros(max(len(a), len(b)) - 1)

# ============================================================
# DATA
# ============================================================

raw_history = collections.deque([0.0]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
env_history = collections.deque([0.0]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

breath_times = []
bpm_history = []

threshold = None
calibration_values = []

is_above = False
phase = "IDLE"

# ============================================================
# BPM FUNCTION
# ============================================================

def compute_bpm():
    now = time.time()
    recent = [t for t in breath_times if now - t < 30]

    if len(recent) < 2:
        return 0

    intervals = np.diff(recent)
    avg = np.mean(intervals)

    return 60 / avg if avg > 0 else 0

# ============================================================
# PLOT
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

line_raw, = ax1.plot([], [])
line_env, = ax2.plot([], [])

threshold_line = ax2.axhline(y=0, linestyle='--')

bpm_text = ax2.text(0.02, 0.9, "", transform=ax2.transAxes)
phase_text = ax1.text(0.02, 0.9, "", transform=ax1.transAxes)

# ============================================================
# UPDATE LOOP
# ============================================================

start_time = time.time()

def update(frame):
    global zi, threshold, is_above, phase

    data = stream.read(CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.float32)

    rms = np.sqrt(np.mean(samples**2))
    raw_history.append(rms)

    filtered, zi = lfilter(b, a, [rms], zi=zi)
    env = abs(filtered[0])
    env_history.append(env)

    # -------- AUTO CALIBRATION --------
    if time.time() - start_time < AUTO_CALIBRATION_TIME:
        calibration_values.append(env)
        phase_text.set_text("CALIBRATING...")
    else:
        if threshold is None:
            threshold = max(calibration_values) * 0.5
            print(f"Auto threshold set: {threshold:.4f}")

    # -------- BREATH DETECTION --------
    if threshold:
        if env > threshold and not is_above:
            is_above = True
            breath_times.append(time.time())

            bpm = compute_bpm()
            bpm_history.append(bpm)

        elif env < threshold * 0.7:
            is_above = False

        # Phase detection
        if env > threshold:
            phase = "INHALE"
        elif is_above:
            phase = "EXHALE"
        else:
            phase = "IDLE"

    # -------- SMOOTH BPM --------
    if bpm_history:
        smoothed_bpm = np.mean(bpm_history[-5:])
    else:
        smoothed_bpm = 0

    # -------- UPDATE PLOT --------
    x = list(range(HISTORY_LENGTH))

    line_raw.set_data(x, raw_history)
    line_env.set_data(x, env_history)

    ax1.set_xlim(0, HISTORY_LENGTH)
    ax2.set_xlim(0, HISTORY_LENGTH)

    ax1.set_ylim(0, max(raw_history)*1.5 + 0.001)
    ax2.set_ylim(0, max(env_history)*1.5 + 0.001)

    if threshold:
        threshold_line.set_ydata([threshold])

    bpm_text.set_text(f"BPM: {smoothed_bpm:.1f}")
    phase_text.set_text(f"Phase: {phase}")

    return line_raw, line_env, bpm_text, phase_text

# ============================================================
# RUN
# ============================================================

ani = animation.FuncAnimation(
    fig, update,
    interval=int(1000 * CHUNK / RATE),
    blit=False
)

plt.tight_layout()
plt.show()

# ============================================================
# CLEANUP + SAVE
# ============================================================

stream.stop_stream()
stream.close()
pa.terminate()

# Save session data
with open("breath_session.txt", "w") as f:
    for bpm in bpm_history:
        f.write(f"{bpm}\n")

print("Session saved → breath_session.txt")