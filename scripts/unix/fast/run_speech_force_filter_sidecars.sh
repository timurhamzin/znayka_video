#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Force rebuild spans and force re-filter sidecar subtitles (video_name.srt).
# Typical duration:
# - ~3-12 seconds per regular episode.
# Expected input:
# - .mp4 and sidecar .srt in the same folder.
# Expected output:
# - New spans + rewritten sidecar .srt filtered by speech.
# - No Whisper, no translation.
#
# Usage:
#   ./scripts/unix/fast/run_speech_force_filter_sidecars.sh [video_folder] [single_video]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

VIDEO_FOLDER="${1:-}"
SINGLE_VIDEO="${2:-}"

source ".venv/bin/activate"

if [[ -n "$VIDEO_FOLDER" ]]; then export TRANSCRIBE_VIDEO_FOLDER="$VIDEO_FOLDER"; fi
export TRANSCRIBE_SINGLE_VIDEO="$SINGLE_VIDEO"
export TRANSCRIBE_ENABLE_SPEECH_SPANS="true"
export TRANSCRIBE_SPEECH_SPANS_ONLY_MODE="true"
export TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS="true"
export TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING="true"
export TRANSCRIBE_SPEECH_SPANS_OVERWRITE="true"
export TRANSCRIBE_OUTPUT_FOLDER=""

python ./transcribe.py

