# When to use:
# - Download videos from DOWNLOAD_URL / PLAYLIST_URL in .env.
# Typical duration:
# - ~1-60+ minutes depending on internet speed, playlist size, and resolution.
# Expected input:
# - .env with DOWNLOAD_URL (or PLAYLIST_URL) and VIDEO_RESOLUTION.
# Expected output:
# - Downloaded .mp4 files in your configured download location.

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
python .\download_video.py

