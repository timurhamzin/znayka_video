param(
    [string]$VideoFolder = "",
    [string]$SingleVideo = ""
)

# When to use:
# - Force rebuild spans and force re-filter sidecar subtitles (video_name.srt).
# Typical duration:
# - ~3-12 seconds per regular episode.
# Expected input:
# - .mp4 and sidecar .srt in the same folder.
# Expected output:
# - New spans + rewritten sidecar .srt filtered by speech.
# - No Whisper, no translation.

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
$env:TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS = "true"
$env:TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING = "true"
$env:TRANSCRIBE_SPEECH_SPANS_OVERWRITE = "true"
$env:TRANSCRIBE_OUTPUT_FOLDER = ""

python .\transcribe.py

