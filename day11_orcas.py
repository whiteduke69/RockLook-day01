# MoodSynth 🎹 day11 of orcas

# AI-powered ambient sound generator.

## How it works
# Text → LLM → Synth Parameters → Audio

## Features
# - Natural language mood input
# - Local LLM (Ollama)
# - Real-time audio synthesis
# - Multiple waveforms (sine, square, triangle, sawtooth, noise)
# - Reverb + tremolo effects

## Run
# ollama serve
# ollama pull qwen2.5:3b




import subprocess
import json
import numpy as np
import sounddevice as sd
import sys
import re
import time

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            print("ERROR: ollama is not running.")
            print("Fix: ollama serve")
            sys.exit(1)
        if "qwen2.5" not in result.stdout.lower():
            print("ERROR: qwen2.5:3b not pulled.")
            print("Fix: ollama pull qwen2.5:3b")
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: ollama not installed.")
        sys.exit(1)

check_ollama()

MODEL = "qwen2.5:3b"
SAMPLE_RATE = 44100

# ============================================================
# 🔥 FIXED PROMPT (VERY STRICT JSON)
# ============================================================

SYNTH_PROMPT_TEMPLATE = """You are a sound synthesis AI.

Convert the given mood into STRICT JSON.

Mood: "{mood}"

RULES:
- OUTPUT ONLY VALID JSON
- NO TEXT BEFORE OR AFTER
- NO MARKDOWN
- NO EXPLANATION
- NO CODE BLOCKS

FORMAT:
{{
  "base_freq": integer (80-800),
  "tempo": float (0.3-3.0),
  "waveform": "sine" | "triangle" | "square" | "sawtooth" | "noise",
  "reverb": float (0.0-1.0),
  "amplitude": float (0.05-0.4),
  "harmonics": integer (1-5)
}}

STYLE MAPPING:
- calm → sine, low freq, slow tempo, high reverb
- tense → square, mid freq, medium tempo
- energetic → sawtooth, high freq, fast tempo
- natural → noise + reverb
- dreamy → triangle + high reverb

RETURN JSON ONLY:"""

# ============================================================
# LLM
# ============================================================

def get_params_from_mood(mood):
    prompt = SYNTH_PROMPT_TEMPLATE.format(mood=mood)

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True, text=True, timeout=30
        )
        raw = result.stdout.strip()
    except Exception:
        return get_default_params()

    json_str = extract_json(raw)

    if not json_str:
        return get_default_params()

    try:
        params = json.loads(json_str)
    except:
        return get_default_params()

    return validate_params(params)


def extract_json(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    return text[start:end + 1]

# ============================================================
# PARAM VALIDATION
# ============================================================

def validate_params(params):
    return {
        "base_freq": int(np.clip(params.get("base_freq", 220), 80, 800)),
        "tempo": float(np.clip(params.get("tempo", 1.0), 0.3, 3.0)),
        "waveform": str(params.get("waveform", "sine")).lower(),
        "reverb": float(np.clip(params.get("reverb", 0.3), 0.0, 1.0)),
        "amplitude": float(np.clip(params.get("amplitude", 0.2), 0.05, 0.4)),
        "harmonics": int(np.clip(params.get("harmonics", 1), 1, 5)),
    }

def get_default_params():
    return {
        "base_freq": 220,
        "tempo": 1.0,
        "waveform": "sine",
        "reverb": 0.3,
        "amplitude": 0.2,
        "harmonics": 1,
    }

# ============================================================
# 🎵 WAVEFORMS
# ============================================================

def gen_sine(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return np.sin(2 * np.pi * freq * t)

def gen_square(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return np.sign(np.sin(2 * np.pi * freq * t))

def gen_triangle(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1

def gen_sawtooth(freq, duration):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return 2 * (t * freq - np.floor(t * freq + 0.5))

# 🔥 NEW: NOISE (BEST ADDITION)
def gen_noise(freq, duration):
    return np.random.uniform(-1, 1, int(SAMPLE_RATE * duration))

WAVEFORMS = {
    "sine": gen_sine,
    "square": gen_square,
    "triangle": gen_triangle,
    "sawtooth": gen_sawtooth,
    "noise": gen_noise,
}

# ============================================================
# EFFECTS
# ============================================================

def apply_reverb(signal, depth):
    if depth <= 0:
        return signal

    output = signal.copy()
    delays = [29, 47, 73]
    for d in delays:
        shift = int(SAMPLE_RATE * d / 1000)
        if shift < len(signal):
            output[shift:] += signal[:-shift] * depth * 0.5

    return output / np.max(np.abs(output))

# 🔥 NEW: TREMOLO EFFECT
def apply_tremolo(signal, rate):
    t = np.linspace(0, len(signal) / SAMPLE_RATE, len(signal))
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)
    return signal * mod

def apply_envelope(signal):
    fade = int(0.05 * SAMPLE_RATE)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    return signal

# ============================================================
# SYNTH
# ============================================================

def synthesize(params, duration=8.0):
    wave_func = WAVEFORMS.get(params["waveform"], gen_sine)

    signal = wave_func(params["base_freq"], duration)

    # harmonics
    for h in range(2, params["harmonics"] + 1):
        signal += wave_func(params["base_freq"] * h, duration) / (h * 2)

    # tremolo = tempo
    signal = apply_tremolo(signal, params["tempo"])

    signal = apply_reverb(signal, params["reverb"])
    signal *= params["amplitude"]

    signal = apply_envelope(signal)

    return np.clip(signal, -1, 1).astype(np.float32)

# ============================================================
# UI
# ============================================================

def main():
    print("\n🎹 MoodSynth READY\n")

    while True:
        mood = input("Mood > ")

        if mood.lower() in ["quit", "exit"]:
            break

        print("⏳ Thinking...")
        params = get_params_from_mood(mood)

        print("⚙️", params)

        audio = synthesize(params)

        print("▶ Playing...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

    print("Done.")

if __name__ == "__main__":
    main()