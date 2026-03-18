#!/usr/bin/env bash
set -euo pipefail

# Master runner (Linux/macOS):
# - Use this for all workflows (spans, filtering, transcription, translation, merge, sidecar replace).
# - Configure behavior in .env via TRANSCRIBE_RUN_* and TRANSCRIBE_FORCE_* flags.
# - Input: folder from TRANSCRIBE_VIDEO_FOLDER (or optional TRANSCRIBE_SINGLE_VIDEO).
# - Output: per-video subtitles/variants + optional merged markdown report.
# - Duration: fast steps (spans/filter/merge) are usually minutes; Whisper transcription can be long.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

source ".venv/bin/activate"
python ./main.py
