$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

& "$PSScriptRoot\check_cuda.ps1" | Out-Null
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml --gpu
