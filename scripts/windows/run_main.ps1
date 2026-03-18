# Master runner (Windows):
# - Use this for all workflows (spans, filtering, transcription, translation, merge, sidecar replace).
# - Configure behavior in .env via TRANSCRIBE_RUN_* and TRANSCRIBE_FORCE_* flags.
# - Input: folder from TRANSCRIBE_VIDEO_FOLDER (or optional TRANSCRIBE_SINGLE_VIDEO).
# - Output: per-video subtitles/variants + optional merged markdown report.
# - Duration: fast steps (spans/filter/merge) are usually minutes; Whisper transcription can be long.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
Set-Location $repoRoot

$activatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    throw "Virtual environment not found at .venv\Scripts\Activate.ps1. Run 'uv sync' first."
}

. $activatePath
python .\main.py
