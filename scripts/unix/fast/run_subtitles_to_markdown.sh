#!/usr/bin/env bash
set -euo pipefail

# When to use:
# - Merge many .srt files into one markdown document.
# Typical duration:
# - ~seconds to a few minutes, based on subtitle count and file size.
# Expected input:
# - .env with TRANSCRIBE_SUBTITLE_SOURCE_DIR and TRANSCRIBE_SUBTITLE_OUTPUT_MD.
# Expected output:
# - One merged markdown file with subtitle text.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv/bin/activate. Run 'uv sync' first." >&2
  exit 1
fi

source ".venv/bin/activate"
python ./subtitles_to_markdown.py

