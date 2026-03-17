param(
    # Optional override for folder with .mp4 files.
    [string]$VideoFolder = "",
    # Optional: exact filename or stem for one video.
    [string]$SingleVideo = "",
    # Optional: true to rebuild existing *.speech_spans.json files.
    [string]$OverwriteSpans = "false"
)

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
$env:TRANSCRIBE_SPEECH_SPANS_ONLY_MODE = "false"
$env:TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS = "true"
$env:TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER = "true"
$env:TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING = "true"
$env:TRANSCRIBE_SPEECH_SPANS_OVERWRITE = $OverwriteSpans

python .\transcribe.py

