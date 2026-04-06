# WhisperDesk — Day 09 (OrcaOS)

# Local real-time speech-to-text system with zero cloud usage.

## Features
# - Offline transcription (faster-whisper)
# - Adaptive silence detection (VAD+)
# - Multi-threaded pipeline
# - Voice commands (stop / clear)
# - Live latency + WPM stats
# - Auto transcript saving

## Hardware Concept
# Implements real-time DSP pipeline:

# MIC → BUFFER → MODEL → TEXT

# Demonstrates latency vs accuracy tradeoffs identical to embedded systems.

## Run
# pip install faster-whisper pyaudio numpy
# python day09_whisperdesk.py



import pyaudio
import numpy as np
import wave
import tempfile
import os
import sys
import time
import threading
from collections import deque

# ==============================
# CONFIG
# ==============================

RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

RECORD_SECONDS = 3
SILENCE_THRESHOLD = 500

TRANSCRIPT_FILE = "transcript.txt"

# ==============================
# BACKEND (faster-whisper)
# ==============================

try:
    from faster_whisper import WhisperModel
    print("Loading Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    BACKEND = "whisper"
except:
    print("ERROR: Install faster-whisper")
    sys.exit(1)

# ==============================
# GLOBAL STATE
# ==============================

audio_queue = deque()
running = True
full_transcript = []
energy_history = deque(maxlen=10)

# ==============================
# AUDIO UTILS
# ==============================

def compute_rms(audio_data):
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    return np.sqrt(np.mean(samples ** 2))


def is_silent(audio_data):
    rms = compute_rms(audio_data)
    energy_history.append(rms)

    adaptive_threshold = max(SILENCE_THRESHOLD, np.mean(energy_history) * 0.6)
    return rms < adaptive_threshold


def save_audio(audio_data):
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
    return temp.name

# ==============================
# TRANSCRIPTION
# ==============================

def transcribe(path):
    segments, _ = model.transcribe(
        path,
        beam_size=1,
        vad_filter=True
    )
    return " ".join(s.text for s in segments).strip()

# ==============================
# RECORD THREAD
# ==============================

def record_loop(stream):
    global running

    while running:
        frames = []

        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        audio = b''.join(frames)

        if not is_silent(audio):
            audio_queue.append(audio)

# ==============================
# TRANSCRIBE THREAD
# ==============================

def process_loop():
    global running

    while running:
        if not audio_queue:
            time.sleep(0.1)
            continue

        audio = audio_queue.popleft()

        path = save_audio(audio)

        start = time.time()
        text = transcribe(path)
        latency = time.time() - start

        os.unlink(path)

        if text:
            print(f"\n📝 {text}")

            # WPM estimation
            words = len(text.split())
            wpm = int(words / (RECORD_SECONDS / 60))

            print(f"⚡ latency: {latency:.2f}s | WPM: {wpm}")

            full_transcript.append(text)

            # Save to file
            with open(TRANSCRIPT_FILE, "a") as f:
                f.write(text + "\n")

            # Voice commands
            if "stop listening" in text.lower():
                print("🛑 Voice command detected: STOP")
                running = False

            if "clear transcript" in text.lower():
                full_transcript.clear()
                open(TRANSCRIPT_FILE, "w").close()
                print("🧹 Transcript cleared")

# ==============================
# MAIN
# ==============================

def main():
    global running

    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("\n" + "="*50)
    print("🎙️ WhisperDesk PRO")
    print("Say 'stop listening' to exit")
    print("="*50)

    t1 = threading.Thread(target=record_loop, args=(stream,))
    t2 = threading.Thread(target=process_loop)

    t1.start()
    t2.start()

    try:
        while running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        running = False

    t1.join()
    t2.join()

    stream.stop_stream()
    stream.close()
    pa.terminate()

    print("\n📋 Final Transcript:")
    print(" ".join(full_transcript))


if __name__ == "__main__":
    main()