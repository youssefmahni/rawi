#!/usr/bin/env bash
# Idempotent local install for Rawi on the mini PC. No sudo. Run as the demo user:
#   bash ~/rawi/deploy/install.sh
# It never creates or copies the DeepSeek key; the user creates ~/rawi/.env themselves.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

export TMPDIR="${TMPDIR:-$HOME/.cache/pip-tmp}"
mkdir -p "$TMPDIR"

# 1. venv
if [ ! -x .venv/bin/python ]; then
  echo "Creating .venv…"
  python3 -m venv .venv
fi

# 2. Python deps (small /tmp → install temp lives under ~/.cache)
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

# 3. Piper voices (English + the Arabic fallback voice)
mkdir -p voices
dl() { # dl <repo-path> <dest>
  if [ ! -s "$2" ]; then
    echo "Downloading $2…"
    curl -sL -o "$2" "https://huggingface.co/rhasspy/piper-voices/resolve/main/$1"
  fi
}
dl en/en_US/lessac/medium/en_US-lessac-medium.onnx        voices/en_US-lessac-medium.onnx
dl en/en_US/lessac/medium/en_US-lessac-medium.onnx.json   voices/en_US-lessac-medium.onnx.json
dl ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx        voices/ar_JO-kareem-medium.onnx
dl ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json   voices/ar_JO-kareem-medium.onnx.json

# 4. voices.json (from the committed example)
if [ ! -f voices/voices.json ]; then
  cp voices.example.json voices/voices.json
  echo "Created voices/voices.json"
fi

# 5. Warm up the speech-to-text model (downloads it into the HF cache on first run)
.venv/bin/python -c "import stt; stt.get_transcriber()" >/dev/null 2>&1 || true

echo
echo "Install done. Next (manual): create ~/rawi/.env with DEEPSEEK_API_KEY=… and start the app."
