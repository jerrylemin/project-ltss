$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$cudaPath = [Environment]::GetEnvironmentVariable("CUDA_PATH", "Machine")
if (-not $cudaPath) {
    $cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaRoot) {
        $cudaPath = Get-ChildItem -LiteralPath $cudaRoot -Directory |
            Where-Object { $_.Name -like "v*" } |
            Sort-Object Name -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }
}
if ($cudaPath) {
    $env:CUDA_PATH = $cudaPath
    $env:Path = "$cudaPath\bin;$cudaPath\bin\x64;$env:Path"
}

function Convert-ToAuditSafeText {
    param([string]$Text)
    if ($null -eq $Text) { return $Text }
    $safe = $Text
    $repoRootEscaped = [Regex]::Escape($repoRoot)
    $safe = [Regex]::Replace($safe, $repoRootEscaped, "<repo>")
    if ($env:USERPROFILE) {
        $safe = [Regex]::Replace($safe, [Regex]::Escape($env:USERPROFILE), "%USERPROFILE%")
    }
    if ($env:LOCALAPPDATA) {
        $safe = [Regex]::Replace($safe, [Regex]::Escape($env:LOCALAPPDATA), "%LOCALAPPDATA%")
    }
    if ($env:APPDATA) {
        $safe = [Regex]::Replace($safe, [Regex]::Escape($env:APPDATA), "%APPDATA%")
    }
    return $safe
}

function Invoke-CheckCommand {
    param([string]$Command)
    try {
        $output = Invoke-Expression $Command 2>&1 | Out-String
        return @{ command = $Command; exit_code = $LASTEXITCODE; output = (Convert-ToAuditSafeText $output.Trim()) }
    } catch {
        return @{ command = $Command; exit_code = -1; output = (Convert-ToAuditSafeText $_.Exception.Message) }
    }
}

$checks = @(
    (Invoke-CheckCommand "nvidia-smi"),
    (Invoke-CheckCommand "nvcc -V"),
    (Invoke-CheckCommand "where.exe nvcc"),
    (Invoke-CheckCommand "cmd /c echo %CUDA_PATH%"),
    (Invoke-CheckCommand ".\.venv\Scripts\python.exe -c `"import numba; print(numba.__version__)`""),
    (Invoke-CheckCommand ".\.venv\Scripts\python.exe -c `"from numba import cuda; print(cuda.is_available()); print(cuda.gpus)`""),
    (Invoke-CheckCommand ".\.venv\Scripts\python.exe -c `"import sys; print(sys.version)`""),
    (Invoke-CheckCommand ".\.venv\Scripts\python.exe -m pip show numba"),
    (Invoke-CheckCommand ".\.venv\Scripts\python.exe -m pip show numba-cuda")
)

$statusJson = Convert-ToAuditSafeText (.\.venv\Scripts\python.exe -c "import json; from src.gpu.cuda_utils import cuda_status; print(json.dumps(cuda_status(), indent=2))")
$payload = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    cuda_path = $env:CUDA_PATH
    path_cuda_entries = @($env:Path -split ";" | Where-Object { $_ -match "CUDA|NVIDIA GPU Computing Toolkit" })
    checks = $checks
    cuda_status = ($statusJson | ConvertFrom-Json)
}

New-Item -ItemType Directory -Force -Path artifacts | Out-Null
$payload | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 artifacts/environment_check.json
$payload | ConvertTo-Json -Depth 8
