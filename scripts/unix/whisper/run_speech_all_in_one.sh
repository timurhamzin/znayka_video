#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - One-pass full workflow for fresh inputs:
#   span detection -> sidecar filtering -> span-aware Whisper -> translation.
# Typical duration:
# - ~5-60+ minutes per video (can be longer for long videos).
# Expected input:
# - Video folder with .mp4 files; sidecar .srt optional.
# Expected output:
# - New/updated *.speech_spans.json, filtered sidecar .srt,
#   original/translated subtitle outputs, and logs.
#
# Usage:
#   ./scripts/unix/whisper/run_speech_all_in_one.sh [video_folder] [single_video] [overwrite_spans]

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
export TRANSCRIBE_SPEECH_SPANS_ONLY_MODE="false"
export TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS="true"
export TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER="true"
export TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING="true"
export TRANSCRIBE_SPEECH_SPANS_OVERWRITE="$OVERWRITE_SPANS"

python ./transcribe.py

