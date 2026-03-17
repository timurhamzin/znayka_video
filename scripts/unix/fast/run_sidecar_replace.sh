#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Replace sidecar subtitles (<video>.srt) from one variant in <video>/<variant>/<video>.srt.
# Typical duration:
# - ~seconds to a minute.
# Expected input:
# - Existing per-video variant subtitle folders and files.
# Expected output:
# - Sidecar .srt files replaced next to videos.
#
# Usage:
#   ./scripts/unix/fast/run_sidecar_replace.sh [video_folder] [single_video] [variant]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

VIDEO_FOLDER="${1:-}"
SINGLE_VIDEO="${2:-}"
VARIANT="${3:-translated_utf8}"

source ".venv/bin/activate"

if [[ -n "$VIDEO_FOLDER" ]]; then export TRANSCRIBE_VIDEO_FOLDER="$VIDEO_FOLDER"; fi
export TRANSCRIBE_SINGLE_VIDEO="$SINGLE_VIDEO"
export TRANSCRIBE_SIDECAR_REPLACE_VARIANT="$VARIANT"

python ./sidecar_replace.py

