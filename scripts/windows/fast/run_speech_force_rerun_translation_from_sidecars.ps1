param(
    [string]$VideoFolder = "",
    [string]$SingleVideo = ""
)

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

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TRANSCRIBE_WHISPER_LANGUAGE= "en"
$env:TRANSCRIBE_TRANSLATION_SOURCE_LANGUAGE= "en"
$env:TRANSCRIBE_TRANSLATION_TARGET_LANGUAGE= "ru"

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
$env:TRANSCRIBE_ENABLE_SPEECH_SPANS = "false"
$env:TRANSCRIBE_SPEECH_SPANS_ONLY_MODE = "false"
$env:TRANSCRIBE_TRANSLATION_ONLY_MODE = "true"
$env:TRANSCRIBE_TRANSLATION_INPUT = "sidecar"
$env:TRANSCRIBE_TRANSLATION_OVERWRITE = "true"
$env:TRANSCRIBE_TRANSLATION_APPEND_SOURCE = "true"
$env:TRANSCRIBE_UPDATE_SIDECAR_FROM_TRANSLATION = "false"
$env:TRANSCRIBE_OUTPUT_FOLDER = ""

python .\transcribe.py
