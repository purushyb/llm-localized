# LLM Localized

Local AI development environment setup — Ollama + Open WebUI + Open Code.

## Pinned Versions

| Tool | Version | Install method |
|------|---------|---------------|
| **Python** | `3.11.15` | pyenv (auto-installed if missing) |
| **pip** | `24.3.1` | pip self-upgrade |
| **Ollama** | latest | Homebrew / curl |
| **Open WebUI** | `0.9.6` | pip (in venv) |
| **Open Code** | latest | npm (`opencode-ai`) |

> To change versions, edit the `Pinned Versions` section at the top of `setup.sh`.

## Prerequisites

- **pyenv** — recommended for managing Python versions (`brew install pyenv`)
- **Homebrew** — for installing Ollama (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- **npm / Node.js** — for Open Code (`brew install node`)

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
ollama pull llama3.1

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Launch Open WebUI (accessible at http://localhost:8080)
open-webui serve

# 5. Launch Open Code in your project directory
opencode
```

## What the script does

1. **Checks for pyenv** — if found, ensures Python `3.11.15` is installed and uses it to create the venv. If pyenv is missing, falls back to system `python3` but will fail if it's not 3.11 or 3.12.
2. **Creates `.venv`** — idempotent; skips if it already exists with the correct Python version, or recreates if the version is wrong.
3. **Installs Ollama** — via Homebrew (preferred) or the official curl installer.
4. **Installs Open WebUI** — pinned to v0.9.6 inside the venv.
5. **Installs Open Code** — the `opencode-ai` npm package globally.
6. **Generates `requirements.txt`** — a full pip freeze for reproducibility.

## File Structure

```
llm-localized/
├── setup.sh              # Main setup script
├── .venv/                # Python virtual environment (created by script)
├── .python-version       # pyenv local version (created by script)
├── requirements.txt      # Frozen pip dependencies (created by script)
└── README.md             # This file
```
# llm-localized
