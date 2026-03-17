#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Download videos from DOWNLOAD_URL / PLAYLIST_URL in .env.
# Typical duration:
# - ~1-60+ minutes depending on internet speed, playlist size, and resolution.
# Expected input:
# - .env with DOWNLOAD_URL (or PLAYLIST_URL) and VIDEO_RESOLUTION.
# Expected output:
# - Downloaded .mp4 files in your configured download location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

source ".venv/bin/activate"
python ./download_video.py

