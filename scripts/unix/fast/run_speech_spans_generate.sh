#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Build speech span files only (*.speech_spans.json).
# Typical duration:
# - ~3-10 seconds per 5-6 minute episode; ~20-40+ seconds for long episodes.
# Expected input:
# - Video folder with .mp4 files (from .env or argument 1).
# Expected output:
# - One *.speech_spans.json next to each processed .mp4.
# - No subtitle changes, no Whisper run, no translation output.
#
# Usage:
#   ./scripts/unix/fast/run_speech_spans_generate.sh [video_folder] [single_video] [overwrite_spans]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

VIDEO_FOLDER="${1:-}"
SINGLE_VIDEO="${2:-}"
OVERWRITE_SPANS="${3:-false}"

source ".venv/bin/activate"

if [[ -n "$VIDEO_FOLDER" ]]; then export TRANSCRIBE_VIDEO_FOLDER="$VIDEO_FOLDER"; fi
export TRANSCRIBE_SINGLE_VIDEO="$SINGLE_VIDEO"
export TRANSCRIBE_ENABLE_SPEECH_SPANS="true"
export TRANSCRIBE_SPEECH_SPANS_ONLY_MODE="true"
export TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS="false"
export TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING="true"
export TRANSCRIBE_SPEECH_SPANS_OVERWRITE="$OVERWRITE_SPANS"
export TRANSCRIBE_OUTPUT_FOLDER=""

python ./transcribe.py

