#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Force rerun translation only, using filtered sidecar SRT files (video_name.srt).
# - Independent from Whisper (no transcription run).
# Typical duration:
# - ~seconds to a few minutes per video.
# Expected input:
# - Existing sidecar SRT files next to .mp4 files.
# Expected output:
# - Rebuilt translated outputs in <video_folder>/<video_stem>/translated_*.
# - Each translated cue contains original line + translated line.
#
# Usage:
#   ./scripts/unix/fast/run_speech_force_rerun_translation_from_sidecars.sh [video_folder] [single_video]

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
export TRANSCRIBE_ENABLE_SPEECH_SPANS="false"
export TRANSCRIBE_SPEECH_SPANS_ONLY_MODE="false"
export TRANSCRIBE_TRANSLATION_ONLY_MODE="true"
export TRANSCRIBE_TRANSLATION_INPUT="sidecar"
export TRANSCRIBE_TRANSLATION_OVERWRITE="true"
export TRANSCRIBE_TRANSLATION_APPEND_SOURCE="true"
export TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION="false"
export TRANSCRIBE_OUTPUT_FOLDER=""

python ./transcribe.py

