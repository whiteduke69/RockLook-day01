"""
==========================================================
TerminalBrain — Day 10 (Advanced)
==========================================================

TerminalBrain wraps a terminal command, captures stdout/stderr
in real time, detects errors, and uses a local LLM (Ollama)
to suggest fixes instantly.

FEATURES:
- Live stdout/stderr streaming
- Intelligent error detection (regex-based)
- AI-powered fix suggestions (local LLM)
- Error caching (avoids repeated LLM calls)
- Cross-platform support (Windows/Linux/macOS)

TECH STACK:
- Python
- subprocess / threading
- Ollama (qwen2.5:3b)

USAGE:
    python day10_terminalbrain.py <command>

EXAMPLES:
    python day10_terminalbrain.py python -c "import nonexistent_module"
    python day10_terminalbrain.py python -c "print(undefined_var)"
    python day10_terminalbrain.py ls /nonexistent

HARDWARE CONCEPT:
Inspired by watchdog timers:
- Wrapper = watchdog
- Error = interrupt
- LLM = recovery handler

REQUIREMENTS:
- Install Ollama → https://ollama.com
- Run: ollama serve
- Pull model: ollama pull qwen2.5:3b
==========================================================
"""

import sys
import platform
import subprocess
import threading
import queue
import re
import argparse

# ============================================================
# PLATFORM CHECK
# ============================================================

IS_WINDOWS = platform.system() == "Windows"

# ============================================================
# COLORS
# ============================================================

class Color:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    DIM = "\033[2m"
    BOLD = "\033[1m"


def color_text(text, color):
    return f"{color}{text}{Color.RESET}"


# ============================================================
# OLLAMA
# ============================================================

MODEL = "qwen2.5:3b"

def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False, "Run: ollama serve"
        if "qwen2.5" not in result.stdout.lower():
            return False, "Run: ollama pull qwen2.5:3b"
        return True, "ok"
    except FileNotFoundError:
        return False, "Install Ollama from https://ollama.com"
    except Exception as e:
        return False, str(e)


def build_llm_prompt(error_text):
    return f"""
You are an expert terminal debugger.

Error:
{error_text}

Your task:
- Give ONE specific fix
- If missing package → give pip install command
- If syntax → show corrected code
- If command error → give correct command

Rules:
- Max 2 lines
- No explanations
- Only the fix

Answer:
"""


def ask_llm_for_fix(error_text):
    prompt = build_llm_prompt(error_text)

    try:
        result = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=25
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[LLM timeout]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ============================================================
# ERROR DETECTION
# ============================================================

ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"\bError\b:",
    r"\bException\b:",
    r"ModuleNotFoundError",
    r"ImportError",
    r"NameError",
    r"SyntaxError",
    r"TypeError",
    r"ValueError",
    r"KeyError",
    r"AttributeError",
    r"FileNotFoundError",
    r"ZeroDivisionError",

    r"Could not find a version",
    r"No matching distribution found",

    r"command not found",
    r"No such file or directory",
    r"Permission denied",
    r"is not recognized",

    r"FAILED",
    r"FATAL",
    r"CRITICAL",
]

ERROR_REGEX = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)


def is_error_line(line):
    return bool(line and ERROR_REGEX.search(line))


# ============================================================
# CACHE
# ============================================================

fix_cache = {}

def extract_error_signature(error_text):
    for line in error_text.splitlines():
        if "Error" in line or "Exception" in line:
            return line.strip()
    return error_text[:100]


def get_cached_fix(error_text):
    return fix_cache.get(extract_error_signature(error_text))


def cache_fix(error_text, fix):
    fix_cache[extract_error_signature(error_text)] = fix


# ============================================================
# THREAD READER
# ============================================================

def reader_thread(stream, q, name):
    for line in iter(stream.readline, ''):
        if not line:
            break
        q.put((name, line))
    stream.close()


# ============================================================
# GLOBAL STATS
# ============================================================

llm_calls = 0
cache_hits = 0


# ============================================================
# ERROR HANDLER
# ============================================================

def handle_error_block(lines):
    global llm_calls, cache_hits

    error_text = "".join(lines).strip()
    if not error_text:
        return

    cached = get_cached_fix(error_text)

    if cached:
        cache_hits += 1
        print(color_text(f"\n🧠 Cached Fix: {cached}\n", Color.CYAN))
        return

    print(color_text("\n🧠 Thinking...\n", Color.CYAN))

    fix = ask_llm_for_fix(error_text)
    llm_calls += 1

    print(color_text(f"🧠 Fix: {fix}\n", Color.CYAN))

    cache_fix(error_text, fix)


# ============================================================
# MAIN RUNNER
# ============================================================

def run_with_brain(command):
    print(color_text("\nTerminalBrain Running:", Color.CYAN), " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    q = queue.Queue()

    threading.Thread(target=reader_thread, args=(process.stdout, q, "stdout"), daemon=True).start()
    threading.Thread(target=reader_thread, args=(process.stderr, q, "stderr"), daemon=True).start()

    error_buffer = []
    in_error = False
    error_count = 0

    while True:
        try:
            name, line = q.get(timeout=0.1)
        except queue.Empty:
            if process.poll() is not None:
                break
            continue

        if name == "stdout":
            print(color_text(line.rstrip(), Color.WHITE))

            if in_error and error_buffer:
                handle_error_block(error_buffer)
                error_buffer = []
                in_error = False

        else:
            print(color_text(line.rstrip(), Color.RED))

            if is_error_line(line):
                in_error = True
                error_count += 1

            if in_error:
                error_buffer.append(line)

    if error_buffer:
        handle_error_block(error_buffer)

    print(color_text("\n--- SUMMARY ---", Color.DIM))
    print(color_text(f"Exit code: {process.returncode}", Color.GREEN if process.returncode == 0 else Color.RED))
    print(color_text(f"Errors: {error_count}", Color.DIM))
    print(color_text(f"LLM calls: {llm_calls}", Color.DIM))
    print(color_text(f"Cache hits: {cache_hits}", Color.DIM))


# ============================================================
# ENTRY
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()

    ok, msg = check_ollama()
    if not ok:
        print(color_text(msg, Color.RED))
        sys.exit(1)

    print(color_text("✓ Ollama Ready\n", Color.GREEN))

    run_with_brain(args.command)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python day10_terminalbrain.py <command>")
        sys.exit(0)

    main()





