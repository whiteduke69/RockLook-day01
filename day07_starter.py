import numpy as np
import sounddevice as sd
import pygame
import threading

# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
BASE_FREQ = 220.0  # A3
volume = 0.2

active_keys = set()
phases = {}
lock = threading.Lock()

wave_type = "sine"  # sine / square / saw

# ============================================================
# KEY → FREQUENCY (FULL KEYBOARD)
# ============================================================

def key_to_freq(key):
    # map key to semitone offset
    return BASE_FREQ * (2 ** (key / 12))


# ============================================================
# WAVE GENERATORS
# ============================================================

def generate_wave(freq, t, phase):
    if wave_type == "sine":
        return np.sin(2*np.pi*freq*t + phase)
    elif wave_type == "square":
        return np.sign(np.sin(2*np.pi*freq*t + phase))
    elif wave_type == "saw":
        return 2*(t*freq - np.floor(0.5 + t*freq))


# ============================================================
# AUDIO CALLBACK
# ============================================================

def audio_callback(outdata, frames, time, status):
    t = np.arange(frames) / SAMPLE_RATE
    signal = np.zeros(frames, dtype=np.float32)

    with lock:
        keys = list(active_keys)

    for k in keys:
        freq = key_to_freq(k)

        if k not in phases:
            phases[k] = 0.0

        phase = phases[k]

        wave = generate_wave(freq, t, phase)
        signal += wave * volume

        phases[k] += 2*np.pi*freq*frames/SAMPLE_RATE

    # cleanup
    for k in list(phases.keys()):
        if k not in keys:
            del phases[k]

    signal = np.clip(signal, -1, 1)
    outdata[:, 0] = signal


# ============================================================
# PYGAME SETUP
# ============================================================

pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("OrcaSynth v1")
clock = pygame.time.Clock()

font = pygame.font.SysFont("monospace", 16)
big = pygame.font.SysFont("monospace", 26, bold=True)

# ============================================================
# START AUDIO (SAFE)
# ============================================================

stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BUFFER_SIZE,
    channels=1,
    callback=audio_callback
)
stream.start()

# ============================================================
# DRAW OSCILLOSCOPE
# ============================================================

def draw_wave(surface, keys):
    mid = HEIGHT//2
    scale = 120

    n = WIDTH
    t = np.linspace(0, 0.02, n)

    wave = np.zeros(n)

    for k in keys:
        freq = key_to_freq(k)
        wave += np.sin(2*np.pi*freq*t)

    if len(keys) > 0:
        wave /= len(keys)

    pts = []
    for x in range(n):
        y = int(mid - wave[x]*scale)
        pts.append((x, y))

    if len(pts) > 1:
        pygame.draw.lines(surface, (255,255,255), False, pts, 2)


# ============================================================
# MAIN LOOP
# ============================================================

running = True

while running:
    for e in pygame.event.get():

        if e.type == pygame.QUIT:
            running = False

        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False

            elif e.key == pygame.K_1:
                wave_type = "sine"
            elif e.key == pygame.K_2:
                wave_type = "square"
            elif e.key == pygame.K_3:
                wave_type = "saw"

            elif e.key == pygame.K_UP:
                volume = min(1.0, volume + 0.05)
            elif e.key == pygame.K_DOWN:
                volume = max(0.01, volume - 0.05)

            else:
                with lock:
                    active_keys.add(e.key)

        elif e.type == pygame.KEYUP:
            with lock:
                active_keys.discard(e.key)

    screen.fill((10,10,20))

    title = big.render("OrcaSynth v1", True, (255,255,255))
    screen.blit(title, (20, 20))

    info = font.render(f"Wave: {wave_type} | Volume: {volume:.2f} | Keys: {len(active_keys)}", True, (180,180,200))
    screen.blit(info, (20, 60))

    with lock:
        snapshot = set(active_keys)

    draw_wave(screen, snapshot)

    pygame.display.flip()
    clock.tick(60)

# cleanup
stream.stop()
stream.close()
pygame.quit()