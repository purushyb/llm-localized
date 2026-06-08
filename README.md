# LLM Localized

Local AI development environment setup — Ollama + Open WebUI + Open Code + Benchmarks.

## Pinned Versions

| Tool | Version | Install method |
|------|---------|---------------|
| **Python** | `3.11.15` | pyenv (auto-installed if missing) |
| **pip** | `24.3.1` | pip self-upgrade |
| **Ollama** | latest | Homebrew / curl |
| **Open WebUI** | `0.9.6` | pip (in venv) |
| **Open Code** | `0.1.0` | pip (`opencode-ai`, in venv) |

> To change versions, edit the `Pinned Versions` section at the top of `setup.sh`.

## Prerequisites

- **pyenv** — recommended for managing Python versions (`brew install pyenv`)
- **Homebrew** — for installing Ollama

## Quick Start

```bash
# Make the script executable and run it
chmod +x setup.sh
./setup.sh
```

## After Setup

```bash
# 1. Start Ollama server
ollama serve

# 2. Pull a model
ollama pull gemma4:e4b

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Launch Open WebUI (accessible at http://localhost:8080)
open-webui serve

# 5. Launch Open Code in your project directory
opencode
```

---

## 🧪 Benchmark Suite

Benchmark your local Ollama models across **coding**, **text prompting**, and **vision** capabilities.

### Configure Models

Edit `models.json` to list the Ollama models you want to benchmark:

```json
{
    "models": [
        "gemma4:e4b"
    ],
    "settings": {
        "ollama_base_url": "http://localhost:11434",
        "timeout_seconds": 120,
        "temperature": 0.1,
        "num_predict": 1024
    }
}
```

> Only models that are already pulled locally will be tested. Unpulled models are skipped with a warning.

### Run Benchmarks

```bash
# Activate venv first
source .venv/bin/activate

# Run all benchmarks on all models
python benchmark.py

# Test specific models only
python benchmark.py --models gemma4:e4b

# Run only coding tests
python benchmark.py --category coding

# Run only text prompting tests
python benchmark.py --category text

# Run only vision tests
python benchmark.py --category vision

# Export results to JSON
python benchmark.py --export results.json

# Disable colored output (for piping/logging)
python benchmark.py --no-color > log.txt
```

### Test Categories

#### Coding (5 tests)
Each test asks the model to write a Python function, then **executes it** against assertions:

| Test | What it checks |
|------|---------------|
| Reverse a String | Basic string manipulation (no slicing) |
| Fibonacci Number | Recursion / iteration |
| Prime Check | Edge cases (0, 1, negatives) |
| Two Sum | Hash map / array indexing |
| Flatten Nested List | Recursive data structure handling |

#### Text Prompting (5 tests)
Validates responses with heuristic checks (keywords, format, correctness):

| Test | What it checks |
|------|---------------|
| Summarization | Conciseness + key concept retention |
| Logical Reasoning | Combinatorics puzzle (handshake problem) |
| Instruction Following | Exact format compliance (numbered list) |
| Factual Knowledge | Simple factual recall (chemical symbol) |
| Math Word Problem | Multi-step arithmetic |

#### Vision (3 tests) — *auto-skipped for non-vision models*
Generates test images via Pillow and validates model descriptions:

| Test | What it checks |
|------|---------------|
| Shape Recognition | Identify shapes and colors |
| Text Reading (OCR) | Read text from an image |
| Object Counting | Count objects accurately |

### Sample Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊  BENCHMARK RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── Scoreboard ──

  Rank  Model         Score        Time    Tokens
  ─────────────────────────────────────────────────
  🥇    gemma4:e4b    11/13 (85%)   62.3s     5240
```

---

## File Structure

```
llm-localized/
├── setup.sh              # Main setup script
├── benchmark.py          # Benchmark suite
├── models.json           # Models to benchmark (edit this!)
├── test_images/          # Auto-generated vision test images
├── .venv/                # Python virtual environment (created by setup)
├── .python-version       # pyenv local version (created by setup)
├── requirements.txt      # Frozen pip dependencies (created by setup)
└── README.md             # This file
```
