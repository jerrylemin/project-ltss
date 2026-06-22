# SNAP Dataset Setup

The project now benchmarks the five required real SNAP graphs by default. Synthetic graphs remain available for smoke tests.

## Supported Local Filenames

The required normalized files live under `data/graphs/`:

- `data/graphs/roadNet-CA.tsv`
- `data/graphs/com-youtube.tsv`
- `data/graphs/wiki-talk.tsv`
- `data/graphs/amazon0601.tsv`
- `data/graphs/soc-livejournal.tsv`

The loader accepts blank lines, comment lines starting with `#`, whitespace-separated files, and CSV-style comma-separated edge lists. Self-loops and duplicate edges are preserved because they are part of the PageRank graph semantics.

## SNAP Sources

- com-Youtube: https://snap.stanford.edu/data/com-Youtube.html
- roadNet-CA: https://snap.stanford.edu/data/roadNet-CA.html
- amazon0601: https://snap.stanford.edu/data/amazon0601.html
- wiki-Talk: https://snap.stanford.edu/data/wiki-Talk.html
- soc-LiveJournal1: https://snap.stanford.edu/data/soc-LiveJournal1.html

## Helper Script

Download and normalize all five required SNAP graphs to TSV files under `data/graphs/`:

```powershell
.\.venv\Scripts\python.exe scripts/download_graphs.py
```

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

The legacy sample script does not download large datasets unless `-Download` is provided. The required final benchmark uses `scripts/download_graphs.py`.
