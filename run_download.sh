#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

source ".venv/bin/activate"
python ./download_video.py
