param(
    # Optional override for folder with .mp4 + sidecar .srt files.
    [string]$VideoFolder = "",
    # Optional: exact filename or stem for one video.
    [string]$SingleVideo = "",
    # Optional: true to rebuild existing *.speech_spans.json files.
    [string]$OverwriteSpans = "false"
)

# When to use:
# - Apply speech-span filtering to sidecar subtitles (video_name.srt).
# Typical duration:
# - ~3-12 seconds per regular episode; longer for big files.
# Expected input:
# - .mp4 and sidecar .srt files in the same folder.
# - Existing span files, or they will be created automatically.
# Expected output:
# - Sidecar .srt files rewritten with cues outside speech removed.
# - Updated/new *.speech_spans.json files.
# - No Whisper run, no translation output.

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
$env:TRANSCRIBE_SPEECH_SPANS_OVERWRITE = $OverwriteSpans
$env:TRANSCRIBE_OUTPUT_FOLDER = ""

python .\transcribe.py

