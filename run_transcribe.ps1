$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$activatePath = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "Virtual environment not found at .venv\Scripts\Activate.ps1. Run 'uv sync' first."
}

. $activatePath
python .\transcribe.py
