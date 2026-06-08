#!/usr/bin/env bash
# ==============================================================================
# LLM Localized — Setup Script
# Sets up a local AI development environment on macOS:
#   1. Creates a Python virtual environment (via pyenv Python 3.11)
#   2. Installs Ollama (local LLM runner)
#   3. Installs Open WebUI (chat interface for Ollama)
#   4. Installs Open Code (AI coding assistant)
#
# All versions are pinned for reproducibility. Edit the "Pinned Versions"
# section below to upgrade.
# ==============================================================================

set -euo pipefail

# ── Pinned Versions ──────────────────────────────────────────────────────────
PYTHON_VERSION="3.11.15"          # Required: 3.11 or 3.12 (Open WebUI constraint)
OPEN_WEBUI_VERSION="0.9.6"        # pip: open-webui
OPENCODE_VERSION="latest"         # npm: opencode-ai (use "latest" or pin e.g. "0.1.0")
PIP_VERSION="24.3.1"              # pip itself

# ── Colors & helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${CYAN}ℹ ${NC} $*"; }
success() { echo -e "${GREEN}✔ ${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠ ${NC} $*"; }
fail()    { echo -e "${RED}✖ ${NC} $*"; exit 1; }
header()  { echo -e "\n${BOLD}━━━ $* ━━━${NC}\n"; }

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# ── Pre-flight checks ───────────────────────────────────────────────────────
header "Pre-flight checks"

# Check for pyenv (preferred) or fall back to system python3
if command -v pyenv &>/dev/null; then
    info "Found pyenv"
    HAS_PYENV=true

    # Ensure the required Python version is installed via pyenv
    if pyenv versions --bare | grep -qx "${PYTHON_VERSION}"; then
        info "pyenv has Python ${PYTHON_VERSION}"
    else
        info "Installing Python ${PYTHON_VERSION} via pyenv (this may take a few minutes) ..."
        pyenv install "${PYTHON_VERSION}"
        success "Python ${PYTHON_VERSION} installed via pyenv"
    fi

    # Set local pyenv version for this project
    PYTHON3="$(pyenv prefix ${PYTHON_VERSION})/bin/python3"
    info "Using Python at: ${PYTHON3}"
else
    HAS_PYENV=false
    if command -v python3 &>/dev/null; then
        PYTHON3="python3"
        PY_VERSION=$(python3 --version 2>&1)
        PY_MINOR=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

        if [[ "${PY_MINOR}" != "3.11" && "${PY_MINOR}" != "3.12" ]]; then
            fail "Found ${PY_VERSION} but Open WebUI requires Python 3.11 or 3.12.\n         Install pyenv and run this script again:\n           brew install pyenv\n           pyenv install ${PYTHON_VERSION}"
        fi
        info "Found ${PY_VERSION} ✓"
    else
        fail "Python 3 is required but not found.\n         Install pyenv: brew install pyenv\n         Then: pyenv install ${PYTHON_VERSION}"
    fi
fi

# Check for Homebrew (used for Ollama)
if command -v brew &>/dev/null; then
    info "Found Homebrew"
    HAS_BREW=true
else
    warn "Homebrew not found — will use curl installer for Ollama"
    HAS_BREW=false
fi

# Check for npm (used for opencode-ai)
if command -v npm &>/dev/null; then
    NPM_VERSION=$(npm --version 2>&1)
    info "Found npm v${NPM_VERSION}"
    HAS_NPM=true
else
    warn "npm not found — will skip Open Code installation"
    HAS_NPM=false
fi

# ── Step 1: Create Python virtual environment ───────────────────────────────
header "Step 1 · Python Virtual Environment"

if [ -d "${VENV_DIR}" ]; then
    # Verify existing venv uses the correct Python version
    EXISTING_PY=$("${PYTHON}" --version 2>&1 || echo "unknown")
    if echo "${EXISTING_PY}" | grep -q "${PYTHON_VERSION}"; then
        warn "Virtual environment already exists (${EXISTING_PY})"
        info "Skipping creation (delete .venv to recreate)"
    else
        warn "Existing venv uses ${EXISTING_PY}, but ${PYTHON_VERSION} is required"
        info "Recreating virtual environment ..."
        rm -rf "${VENV_DIR}"
        ${PYTHON3} -m venv "${VENV_DIR}"
        success "Virtual environment recreated with Python ${PYTHON_VERSION}"
    fi
else
    info "Creating virtual environment with Python ${PYTHON_VERSION} ..."
    ${PYTHON3} -m venv "${VENV_DIR}"
    success "Virtual environment created"
fi

# Pin pyenv local version so future shells pick it up
if [ "${HAS_PYENV}" = true ]; then
    echo "${PYTHON_VERSION}" > "${SCRIPT_DIR}/.python-version"
    info "Wrote .python-version (${PYTHON_VERSION})"
fi

# Upgrade pip to pinned version
info "Installing pip==${PIP_VERSION} ..."
"${PIP}" install --upgrade "pip==${PIP_VERSION}" --quiet
success "pip ${PIP_VERSION} installed"

# ── Step 2: Install Ollama ───────────────────────────────────────────────────
header "Step 2 · Ollama"

if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 || true)
    success "Ollama is already installed (${OLLAMA_VERSION})"
else
    info "Installing Ollama ..."
    if [ "${HAS_BREW}" = true ]; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    success "Ollama installed"
fi

info "To start the Ollama server, run:  ${BOLD}ollama serve${NC}"
info "To pull a model, run:             ${BOLD}ollama pull llama3.1${NC}"

# ── Step 3: Install Open WebUI ───────────────────────────────────────────────
header "Step 3 · Open WebUI (v${OPEN_WEBUI_VERSION})"

info "Installing open-webui==${OPEN_WEBUI_VERSION} into the virtual environment ..."
"${PIP}" install "open-webui==${OPEN_WEBUI_VERSION}" --quiet
success "Open WebUI ${OPEN_WEBUI_VERSION} installed"

info "To launch Open WebUI, run:"
echo -e "  ${BOLD}source ${VENV_DIR}/bin/activate${NC}"
echo -e "  ${BOLD}open-webui serve${NC}"

# ── Step 4: Install Open Code ────────────────────────────────────────────────
header "Step 4 · Open Code (opencode-ai@${OPENCODE_VERSION})"

if [ "${HAS_NPM}" = true ]; then
    info "Installing opencode-ai@${OPENCODE_VERSION} via npm ..."
    npm install -g "opencode-ai@${OPENCODE_VERSION}"
    success "Open Code installed"
    info "To launch, run: ${BOLD}opencode${NC}"
else
    warn "npm is not available — skipping Open Code installation"
    info "To install manually later:"
    echo -e "  ${BOLD}npm install -g opencode-ai@${OPENCODE_VERSION}${NC}"
fi

# ── Generate requirements.txt (for reference / reproducibility) ──────────────
header "Generating requirements.txt"

"${PIP}" freeze > "${SCRIPT_DIR}/requirements.txt"
PACKAGE_COUNT=$(wc -l < "${SCRIPT_DIR}/requirements.txt" | tr -d ' ')
success "Wrote requirements.txt (${PACKAGE_COUNT} packages pinned)"

# ── Done ─────────────────────────────────────────────────────────────────────
header "Setup Complete 🎉"

echo -e "Your local AI environment is ready in: ${BOLD}${SCRIPT_DIR}${NC}\n"
echo -e "  ${BOLD}Pinned versions:${NC}"
echo -e "  ${CYAN}•${NC} Python:       ${PYTHON_VERSION}"
echo -e "  ${CYAN}•${NC} Open WebUI:   ${OPEN_WEBUI_VERSION}"
echo -e "  ${CYAN}•${NC} Open Code:    opencode-ai@${OPENCODE_VERSION}"
echo -e "  ${CYAN}•${NC} pip:          ${PIP_VERSION}"
echo ""
echo -e "Quick-start commands:"
echo -e "  ${CYAN}1.${NC} Start Ollama:      ${BOLD}ollama serve${NC}"
echo -e "  ${CYAN}2.${NC} Pull a model:       ${BOLD}ollama pull llama3.1${NC}"
echo -e "  ${CYAN}3.${NC} Activate venv:      ${BOLD}source ${VENV_DIR}/bin/activate${NC}"
echo -e "  ${CYAN}4.${NC} Launch Open WebUI:  ${BOLD}open-webui serve${NC}"
echo -e "  ${CYAN}5.${NC} Launch Open Code:   ${BOLD}opencode${NC}"
echo ""
