# When to use:
# - Merge many .srt files into one markdown document.
# Typical duration:
# - ~seconds to a few minutes, based on subtitle count and file size.
# Expected input:
# - .env with TRANSCRIBE_SUBTITLE_SOURCE_DIR and TRANSCRIBE_SUBTITLE_OUTPUT_MD.
# Expected output:
# - One merged markdown file with subtitle text.

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
python .\subtitles_to_markdown.py

