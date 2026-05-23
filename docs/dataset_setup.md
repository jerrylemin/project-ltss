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

The script does not download large datasets unless `-Download` is provided.
