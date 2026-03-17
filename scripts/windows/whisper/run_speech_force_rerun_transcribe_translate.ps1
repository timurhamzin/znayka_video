param(
    [string]$VideoFolder = "",
    [string]$SingleVideo = ""
)

# When to use:
# - Force rerun of transcription + translation with fresh spans and sidecar filtering.
# Typical duration:
# - ~5-60+ minutes per video (depends on length/hardware).
# Expected input:
# - .mp4 files (sidecar .srt optional, filtered if present).
# Expected output:
# - Recreated per-video output folders (original/translated/stdout).
# - Sidecar filtering applied before transcription.

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
$folder = if ($VideoFolder -ne "") { $VideoFolder } else { $env:TRANSCRIBE_VIDEO_FOLDER }

if (-not $folder) {
    throw "TRANSCRIBE_VIDEO_FOLDER is not set. Pass -VideoFolder or set it in .env."
}

# Force rerun by removing prior per-video outputs (<video_folder>/<video_stem>/...).
$videos = Get-ChildItem -Path $folder -File -Filter *.mp4
if ($SingleVideo -ne "") {
    $target = $SingleVideo.ToLowerInvariant()
    $videos = $videos | Where-Object {
        $_.Name.ToLowerInvariant() -eq $target -or $_.BaseName.ToLowerInvariant() -eq $target
    }
}

foreach ($video in $videos) {
    $outDir = Join-Path $folder $video.BaseName
    if (Test-Path $outDir) {
        Remove-Item -Path $outDir -Recurse -Force
    }
}

$env:TRANSCRIBE_SINGLE_VIDEO = $SingleVideo
$env:TRANSCRIBE_ENABLE_SPEECH_SPANS = "true"
$env:TRANSCRIBE_SPEECH_SPANS_ONLY_MODE = "false"
$env:TRANSCRIBE_FILTER_SIDECAR_SRT_BY_SPEECH_SPANS = "true"
$env:TRANSCRIBE_USE_SPEECH_SPANS_FOR_WHISPER = "true"
$env:TRANSCRIBE_SPEECH_SPANS_DETECT_IF_MISSING = "true"
$env:TRANSCRIBE_SPEECH_SPANS_OVERWRITE = "true"
$env:TRANSCRIBE_OUTPUT_FOLDER = ""

python .\transcribe.py

