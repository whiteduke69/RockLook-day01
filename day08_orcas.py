# PocketAgent — Day 08

# PocketAgent is a local AI agent that runs a 3B-parameter LLM on-device with tool usage, file access, and web browsing capabilities.

# Features
# - CLI-based AI assistant (interactive loop)
# - Runs fully local LLM via Ollama (no API, no cloud)
# - Tool usage system (file system + system info + web)
# - Web browsing (search + fetch content)
# - Smart tool filtering (prevents hallucinated tool calls)
# - Built-in math handling (correct arithmetic)
# - Tokens/sec performance display

# Tech Stack
# - Python
# - Ollama (local LLM runtime)
# - subprocess
# - requests
# - BeautifulSoup
# - argparse (conceptually for CLI structure)

# Controls
# - Type any query → agent responds
# - "exit" / "quit" → stop
# - Ask about files → uses tools
# - Ask general questions → direct answers
# - Ask for news → web search

# Hardware Concept
# - Edge inference (LLM runs locally within RAM constraints)
# - Memory budget (~1.5GB for 3B model with quantization)
# - Tool routing = interrupt handling (input → correct function)
# - Hybrid architecture (rule-based + neural inference)

# Notes
# - Requires ollama serve running in background
# - Model must be pulled (ollama pull qwen2.5:3b)
# - Tool filtering prevents incorrect tool usage
# - Local inference may be slower (~2–10 tokens/sec)
# - Web tools simulate retrieval-augmented generation (RAG-lite)
# - System prompt strongly affects agent reliability


import subprocess
import os
import sys
import time
import platform
import requests
from bs4 import BeautifulSoup

# ============================================================
# CONFIG
# ============================================================

MODEL = "qwen2.5:3b"

# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode != 0:
            print("❌ ollama not running")
            print("Fix: run `ollama serve`")
            sys.exit(1)

        if MODEL not in result.stdout:
            print("❌ model not found")
            print(f"Fix: ollama pull {MODEL}")
            sys.exit(1)

        print("✅ ollama ready")
    except Exception:
        print("❌ ollama not installed")
        sys.exit(1)

check_ollama()

# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(messages):
    prompt = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in messages
    ) + "\nAssistant:"

    start = time.time()

    result = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True, text=True
    )

    response = result.stdout.strip()

    elapsed = time.time() - start
    tokens = len(response) / 4
    tps = tokens / elapsed if elapsed > 0 else 0

    return response, tps

# ============================================================
# TOOLS
# ============================================================

def list_dir(path="."):
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return f"Error: {e}"

def read_file(path):
    try:
        with open(path, "r") as f:
            return f.read(1500)
    except Exception as e:
        return f"Error: {e}"

def system_info():
    return f"""
OS: {platform.system()}
Version: {platform.version()}
CPU: {platform.processor()}
Python: {platform.python_version()}
Directory: {os.getcwd()}
"""

def current_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def find_files(name):
    results = []
    for root, dirs, files in os.walk("."):
        for f in files:
            if name.lower() in f.lower():
                results.append(os.path.join(root, f))
    return "\n".join(results[:20]) or "No files found"

def web_search(query):
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")

        results = []
        for a in soup.select(".result__a")[:5]:
            results.append(a.get_text())

        return "\n".join(results) if results else "No results"
    except Exception as e:
        return f"Error: {e}"

def fetch_url(url):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text() for p in paragraphs[:10])

        return text[:2000] if text else "No content"
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "list_directory": list_dir,
    "read_file": read_file,
    "system_info": system_info,
    "current_time": current_time,
    "find_files": find_files,
    "web_search": web_search,
    "fetch_url": fetch_url,
}

# ============================================================
# TOOL FILTER (IMPORTANT)
# ============================================================

def should_use_tool(user_input):
    keywords = [
        "file", "directory", "list", "read",
        "system", "cpu", "os",
        "time", "date",
        "news", "latest", "search"
    ]
    return any(k in user_input.lower() for k in keywords)

# ============================================================
# SIMPLE MATH FIX
# ============================================================

def try_math(user_input):
    try:
        return str(eval(user_input))
    except:
        return None

# ============================================================
# TOOL PARSER
# ============================================================

def parse_tool(text):
    for line in text.split("\n"):
        if line.startswith("TOOL:"):
            parts = line[5:].strip().split(maxsplit=1)
            name = parts[0]
            arg = parts[1] if len(parts) > 1 else None
            return name, arg
    return None, None

# ============================================================
# SYSTEM PROMPT (FIXED)
# ============================================================

SYSTEM = """
You are PocketAgent, a local AI assistant.

Available tools:
- list_directory [path]
- read_file <file>
- system_info
- current_time
- find_files <name>
- web_search <query>
- fetch_url <url>

STRICT RULES:

1. ONLY use tools for:
   - files, directories, system info
   - real-time or external data (time, news, web)

2. DO NOT use tools for:
   - general knowledge
   - math
   - geography, history, definitions

3. If the question is simple → answer directly

4. If using a tool, respond EXACTLY:
   TOOL: tool_name argument

5. Never invent tool names

Examples:

User: What is the capital of France?
Assistant: Paris

User: 10+20
Assistant: 30

User: List files
Assistant: TOOL: list_directory .

User: Latest AI news
Assistant: TOOL: web_search latest AI news
"""

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("\n🤖 PocketAgent v3 (Stable)\n")

    messages = [{"role": "system", "content": SYSTEM}]

    while True:
        user = input("You > ")

        if user.lower() in ["exit", "quit"]:
            break

        # ✅ math shortcut
        math_result = try_math(user)
        if math_result:
            print("Agent >", math_result)
            continue

        messages.append({"role": "user", "content": user})

        print("⏳ Thinking...")
        response, tps = chat(messages)

        tool, arg = parse_tool(response)

        # 🚨 block bad tool usage
        if tool and not should_use_tool(user):
            tool = None
            arg = None

        if tool and tool in TOOLS:
            print(f"🔧 {tool} {arg or ''}")

            result = TOOLS[tool](arg) if arg else TOOLS[tool]()
            print("\n" + result + "\n")

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{result}\nExplain briefly."
            })

            response, tps = chat(messages)
            print("Agent >", response)

            messages.append({"role": "assistant", "content": response})

        else:
            print("Agent >", response)
            messages.append({"role": "assistant", "content": response})

        print(f"\n⚡ {tps:.1f} tokens/sec\n")

        if len(messages) > 12:
            messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    main()