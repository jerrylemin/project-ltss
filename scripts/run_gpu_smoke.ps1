$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

& "$PSScriptRoot\check_cuda.ps1" | Out-Null
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_optional.py -v
