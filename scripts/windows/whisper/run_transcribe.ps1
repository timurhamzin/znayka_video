# When to use:
# - Run the standard transcription pipeline from .env settings.
# Typical duration:
# - ~5-60+ minutes per video (depends on video length and model speed).
# Expected input:
# - .env with TRANSCRIBE_* variables (video folder, language, translation settings).
# Expected output:
# - Per-video folders with original/translated subtitles and logs.

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
python .\transcribe.py

