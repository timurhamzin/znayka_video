param(
    # Optional override for folder with .mp4 files.
    [string]$VideoFolder = "",
    # Optional: exact filename or stem for one video.
    [string]$SingleVideo = "",
    # Optional: true to rebuild existing *.speech_spans.json files.
    [string]$OverwriteSpans = "false"
)

# When to use:
# - Build speech span files only (*.speech_spans.json).
# Typical duration:
# - ~3-10 seconds per 5-6 minute episode; ~20-40+ seconds for long episodes.
# Expected input:
# - Video folder with .mp4 files (from .env or -VideoFolder).
# Expected output:
# - One *.speech_spans.json next to each processed .mp4.
# - No subtitle changes, no Whisper run, no translation output.

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")
Set-Location $repoRoot

$activatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "Virtual environment not found at .venv\Scripts\Activate.ps1. Run 'uv sync' first."
}

. $activatePath

if ($VideoFolder -ne "") { $env:TRANSCRIBE_VIDEO_FOLDER = $VideoFolder }
$env:TRANSCRIBE_SINGLE_VIDEO = $SingleVideo
$env:TRANSCRIBE_ENABLE_SPEECH_SPANS = "true"
$env:TRANSCRIBE_SPEECH_SPANS_ONLY_MODE = "true"
$env:TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS = "false"
$env:TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING = "true"
$env:TRANSCRIBE_SPEECH_SPANS_OVERWRITE = $OverwriteSpans
$env:TRANSCRIBE_OUTPUT_FOLDER = ""

python .\transcribe.py

