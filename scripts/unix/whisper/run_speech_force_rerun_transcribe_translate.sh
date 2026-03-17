#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Force rerun of transcription + translation with fresh spans and sidecar filtering.
# Typical duration:
# - ~5-60+ minutes per video (depends on length/hardware).
# Expected input:
# - .mp4 files (sidecar .srt optional, filtered if present).
# Expected output:
# - Recreated per-video output folders (original/translated/stdout).
# - Sidecar filtering applied before transcription.
#
# Usage:
#   ./scripts/unix/whisper/run_speech_force_rerun_transcribe_translate.sh [video_folder] [single_video]

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
FOLDER="${TRANSCRIBE_VIDEO_FOLDER:-}"
if [[ -z "$FOLDER" ]]; then
  echo "TRANSCRIBE_VIDEO_FOLDER is not set. Pass [video_folder] or set it in .env." >&2
  exit 1
fi

# Force rerun by removing prior per-video outputs (<video_folder>/<video_stem>/...).
shopt -s nullglob
videos=("$FOLDER"/*.mp4)
for video in "${videos[@]}"; do
  base="$(basename "$video")"
  stem="${base%.*}"
  if [[ -n "$SINGLE_VIDEO" ]]; then
    lc_single="$(printf '%s' "$SINGLE_VIDEO" | tr '[:upper:]' '[:lower:]')"
    lc_base="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
    lc_stem="$(printf '%s' "$stem" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lc_single" != "$lc_base" && "$lc_single" != "$lc_stem" ]]; then
      continue
    fi
  fi
  out_dir="$FOLDER/$stem"
  if [[ -d "$out_dir" ]]; then
    rm -rf "$out_dir"
  fi
done

export TRANSCRIBE_SINGLE_VIDEO="$SINGLE_VIDEO"
export TRANSCRIBE_ENABLE_SPEECH_SPANS="true"
export TRANSCRIBE_SPEECH_SPANS_ONLY_MODE="false"
export TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS="true"
export TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER="true"
export TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING="true"
export TRANSCRIBE_SPEECH_SPANS_OVERWRITE="true"
export TRANSCRIBE_OUTPUT_FOLDER=""

python ./transcribe.py

