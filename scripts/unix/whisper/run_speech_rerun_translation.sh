#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Re-run Whisper + translation with speech spans enabled.
# Typical duration:
# - ~5-60+ minutes per video, depending on length and hardware.
# Expected input:
# - Video folder with .mp4 files.
# - Optional existing span files (or they will be created).
# Expected output:
# - Updated transcription/translation outputs in per-video folders.
# - Sidecar .srt filtered by speech spans.
#
# Usage:
#   ./scripts/unix/whisper/run_speech_rerun_translation.sh [video_folder] [single_video] [overwrite_spans]

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

