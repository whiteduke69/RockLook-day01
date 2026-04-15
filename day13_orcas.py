"""
BUILDCORED ORCAS — Day 13: DailyDebrief++

Upgraded to Hall of Fame level:
- Smarter LLM prompt
- Code-aware analysis (NEW data source)
- Focus score metric
- Pattern detection
- Cleaner UI

Run: python day13.py
"""

import subprocess, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
except ImportError:
    print("pip install rich"); sys.exit(1)

MODEL = "qwen2.5:3b"

# ====== CHECK OLLAMA ======
def check_ollama():
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if "qwen2.5" not in r.stdout.lower():
            console.print("[red]Run: ollama pull qwen2.5:3b[/red]"); sys.exit(1)
    except:
        console.print("[red]ollama not found[/red]"); sys.exit(1)

check_ollama()

# ====== DATA SOURCES ======

def get_git_commits(hours=24):
    try:
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        r = subprocess.run(
            ["git", "log", f"--since={since}", "--pretty=format:%h %s"],
            capture_output=True, text=True, timeout=5
        )
        commits = r.stdout.strip().split("\n") if r.stdout.strip() else []
        return commits[:20]
    except:
        return []

def get_recent_files(hours=24):
    home = Path.home()
    cutoff = time.time() - (hours * 3600)
    recent = []

    for p in home.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime > cutoff:
                if any(x in str(p) for x in [".cache", "node_modules", ".git/", "__pycache__", "Library"]):
                    continue
                recent.append(str(p.relative_to(home)))
                if len(recent) >= 30:
                    break
        except:
            pass

    return recent

def get_shell_history(lines=30):
    for hist_file in [".zsh_history", ".bash_history"]:
        path = Path.home() / hist_file
        if path.exists():
            try:
                with open(path, "r", errors="ignore") as f:
                    all_lines = f.readlines()
                return [l.strip() for l in all_lines[-lines:] if l.strip()]
            except:
                pass
    return []

# 🔥 NEW: CODE-AWARE SOURCE
def get_code_snippets(hours=24):
    home = Path.home()
    cutoff = time.time() - (hours * 3600)
    snippets = []

    for p in home.rglob("*.py"):
        try:
            if p.stat().st_mtime > cutoff:
                if any(x in str(p) for x in [".cache", "node_modules", ".git", "__pycache__"]):
                    continue

                with open(p, "r", errors="ignore") as f:
                    lines = f.readlines()

                snippet = "".join(lines[-20:])
                snippets.append(f"{p.name}:\n{snippet}")

                if len(snippets) >= 5:
                    break
        except:
            pass

    return snippets

# ====== METRICS ======

def compute_focus_score(commits, files, history):
    score = len(commits)*3 + len(files)*1 + len(history)*0.5
    return round(score, 1)

# ====== LLM PROMPT ======

DEBRIEF_PROMPT = """You are an elite engineering manager analyzing a developer's day.

Your job is to reconstruct their work, detect intent, and extract high-signal insights.

Output EXACTLY 5 sections:

BUILT: A concrete artifact or system they worked on (be specific)
BROKE: A real struggle, bug, confusion, or friction point
LEARNED: A non-obvious insight or skill gained
PATTERN: The deeper theme (e.g. "debugging environment issues", "ML experimentation loop")
NEXT: A sharp, high-leverage next action (not generic)

Rules:
- Be specific, not vague
- Infer intelligently (even if data is incomplete)
- No fluff, no repetition
- Each line must feel insightful

Data:
{data}

5 sections only. No preamble:"""

def get_debrief(data_text):
    prompt = DEBRIEF_PROMPT.format(data=data_text[:4000])
    try:
        r = subprocess.run(
            ["ollama", "run", MODEL, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return r.stdout.strip()
    except:
        return "[LLM error]"

# ====== MAIN ======

console.print("\n[bold cyan]📊 DailyDebrief++[/bold cyan]\n")
console.print("[dim]Collecting data from the last 24 hours...[/dim]\n")

commits = get_git_commits()
files = get_recent_files()
history = get_shell_history()
code = get_code_snippets()

console.print(f"  Git commits:    {len(commits)}")
console.print(f"  Recent files:   {len(files)}")
console.print(f"  Shell commands: {len(history)}")
console.print(f"  Code snippets:  {len(code)}")
console.print()

# ====== BUILD DATA ======

data = []

if commits:
    data.append("GIT COMMITS:\n" + "\n".join(commits[:10]))

if files:
    data.append("FILES MODIFIED:\n" + "\n".join(files[:15]))

if history:
    data.append("SHELL HISTORY:\n" + "\n".join(history[-15:]))

if code:
    data.append("CODE CONTEXT:\n" + "\n\n".join(code[:3]))

if not data:
    console.print("[yellow]No data found. Make some activity first![/yellow]")
    sys.exit(0)

combined = "\n\n".join(data)

# ====== METRICS DISPLAY ======

focus = compute_focus_score(commits, files, history)

table = Table(title="⚡ Activity Summary")
table.add_column("Metric")
table.add_column("Value")

table.add_row("Focus Score", str(focus))
table.add_row("Commits", str(len(commits)))
table.add_row("Files", str(len(files)))
table.add_row("Commands", str(len(history)))

console.print(table)
console.print()

# ====== LLM ======

console.print("[dim]Analyzing your day with AI...[/dim]")
start = time.time()

debrief = get_debrief(combined)

elapsed = time.time() - start

# ====== OUTPUT ======

console.print(Panel(debrief, title=f"🧠 AI Debrief ({elapsed:.1f}s)", border_style="cyan"))

# ====== PATTERN DETECTION ======

lower = debrief.lower()

if "debug" in lower or "error" in lower:
    console.print("[magenta]⚠️ Pattern Detected: Debugging Day[/magenta]")
elif "build" in lower or "implement" in lower:
    console.print("[green]🚀 Pattern Detected: Builder Mode[/green]")
elif "learn" in lower:
    console.print("[blue]📚 Pattern Detected: Learning Day[/blue]")

console.print("\n[bold green]See you tomorrow for Day 14![/bold green]\n")