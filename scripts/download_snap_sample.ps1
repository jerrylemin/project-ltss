param(
    [ValidateSet("sample", "com-youtube", "roadNet-CA", "amazon0601")]
    [string]$Graph = "sample",
    [switch]$Download
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$rawDir = Join-Path $repoRoot "data\raw"
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$snap = @{
    "com-youtube" = @{
        Url = "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
        File = "com-youtube.ungraph.txt.gz"
    }
    "roadNet-CA" = @{
        Url = "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
        File = "roadNet-CA.txt.gz"
    }
    "amazon0601" = @{
        Url = "https://snap.stanford.edu/data/amazon0601.txt.gz"
        File = "amazon0601.txt.gz"
    }
}

if ($Graph -eq "sample") {
    $samplePath = Join-Path $rawDir "sample_snap.txt"
    @"
# Tiny SNAP-style sample for smoke tests
100 200
200 300
300 100
300 400
400 200
"@ | Set-Content -Encoding UTF8 -Path $samplePath
    Write-Host "Created $samplePath"
    Write-Host "Run: python src/benchmark.py --config project_spec.yaml --edges-path data/raw/sample_snap.txt"
    exit 0
}

$entry = $snap[$Graph]
$target = Join-Path $rawDir $entry.File
if (-not $Download) {
    Write-Host "SNAP URL: $($entry.Url)"
    Write-Host "Target path: $target"
    Write-Host "Use -Download to fetch it. These files can be large."
    exit 0
}

Invoke-WebRequest -Uri $entry.Url -OutFile $target
Write-Host "Downloaded $target"
Write-Host "Decompress the .gz file before passing it with --edges-path."
