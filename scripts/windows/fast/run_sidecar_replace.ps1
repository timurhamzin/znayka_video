param(
    [string]$VideoFolder = "",
    [string]$SingleVideo = "",
    # One of: original, translated_utf8, translated_windows1251
    [string]$Variant = "original"
)

# When to use:
# - Replace sidecar subtitles (<video>.srt) from one variant in <video>/<variant>/<video>.srt.
# Typical duration:
# - ~seconds to a minute.
# Expected input:
# - Existing per-video variant subtitle folders and files.
# Expected output:
# - Sidecar .srt files replaced next to videos.

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
$env:TRANSCRIBE_SIDECAR_REPLACE_VARIANT = $Variant

python .\sidecar_replace.py

