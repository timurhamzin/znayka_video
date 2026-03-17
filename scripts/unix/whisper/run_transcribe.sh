#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Run the standard transcription pipeline from .env settings.
# Typical duration:
# - ~5-60+ minutes per video (depends on video length and model speed).
# Expected input:
# - .env with TRANSCRIBE_* variables (video folder, language, translation settings).
# Expected output:
# - Per-video folders with original/translated subtitles and logs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

source ".venv/bin/activate"
python ./transcribe.py

