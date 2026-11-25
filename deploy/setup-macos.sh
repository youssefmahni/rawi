#!/usr/bin/env bash
# Rawi setup for macOS (Mac Pro). Run from the extracted bundle directory.
#   tar -xzf rawi-bundle.tar.gz -C rawi && cd rawi && bash deploy/setup-macos.sh
set -euo pipefail

# ---- 1. Homebrew (skip if already present) ----
if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew…"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Apple Silicon puts brew in /opt/homebrew; Intel in /usr/local
  [ -x /opt/homebrew/bin/brew ] && eval "$(/opt/homebrew/bin/brew shellenv)"
  [ -x /usr/local/bin/brew ] && eval "$(/usr/local/bin/brew shellenv)"
fi

# ---- 2. Python 3.13 + ffmpeg ----
echo "Installing Python 3.13 and ffmpeg…"
brew install python@3.13 ffmpeg

PY="$(brew --prefix python@3.13)/bin/python3.13"
echo "Using $($PY --version)"

# ---- 3. venv + Python deps ----
echo "Creating venv and installing deps…"
"$PY" -m venv .venv
mkdir -p ~/.cache/pip-tmp
TMPDIR=~/.cache/pip-tmp .venv/bin/pip install --upgrade pip
TMPDIR=~/.cache/pip-tmp .venv/bin/pip install -r requirements.txt

# ---- 4. Piper voices + speech-to-text warm-up (idempotent) ----
bash deploy/install.sh

# ---- 5. .env ----
if [ ! -f .env ]; then cp .env.example .env; fi
cat <<'EOF'

Done. Next (manual):
  1. nano .env   → set DEEPSEEK_API_KEY=sk-…
  2. .venv/bin/python app.py
  3. open http://localhost:8765

Notes:
  - macOS has no NVIDIA/CUDA, so STT runs on CPU (fast on a Mac Pro's many cores).
    Leave STT_DEVICE=cpu (the default).
  - To auto-start on boot, use launchd (systemd does not exist on macOS).
EOF
