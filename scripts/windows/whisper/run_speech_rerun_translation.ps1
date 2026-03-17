param(
    # Optional override for folder with .mp4 files.
    [string]$VideoFolder = "",
    # Optional: exact filename or stem for one video.
    [string]$SingleVideo = "",
    # Optional: true to rebuild existing *.speech_spans.json files.
    [string]$OverwriteSpans = "false"
)

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

