# SNAP Dataset Setup

The project runs by default on synthetic graphs. Real SNAP datasets are optional for larger demos.

## Supported Local Filenames

Place decompressed edge-list files under `data/raw/`:

- `data/raw/com-youtube.ungraph.txt`
- `data/raw/roadNet-CA.txt`
- `data/raw/amazon0601.txt`

The loader accepts blank lines, comment lines starting with `#`, whitespace-separated files, and CSV-style comma-separated edge lists.

## SNAP Sources

- com-Youtube: https://snap.stanford.edu/data/com-Youtube.html
- roadNet-CA: https://snap.stanford.edu/data/roadNet-CA.html
- amazon0601: https://snap.stanford.edu/data/amazon0601.html

## Helper Script

Create a tiny local sample:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_snap_sample.ps1
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml --edges-path data/raw/sample_snap.txt
```

Print download URL and target path without downloading:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_snap_sample.ps1 -Graph com-youtube
```

Download explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_snap_sample.ps1 -Graph com-youtube -Download
```

Download roadNet-CA and prepare a bounded sample:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_snap_sample.ps1 -Graph roadNet-CA -Download -MaxEdges 200000
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml --edges-path data/raw/roadNet-CA.sample200000.txt --graph-name roadNet-CA-sample200k --gpu --versions v2,v3_pull,v3_push --max-iter 20 --no-scipy-verify --output artifacts/roadnet_ca_sample_benchmarks.csv
```

The script does not download large datasets unless `-Download` is provided.
