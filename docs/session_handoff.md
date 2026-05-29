# Session Handoff

Last updated: 2026-05-29T14:03:47.2264538+07:00

## Environment Used

- Workspace: `D:\project-ltss`
- Python executable: `D:\project-ltss\.venv\Scripts\python.exe`
- Python: `3.11.9`
- Packages: `numpy==2.4.6`, `scipy==1.17.1`, `numba==0.65.1`, `numba-cuda==0.30.2`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, driver `591.86`, 6144 MiB VRAM
- CUDA availability: `cuda.is_available() == True`

## Dataset Files Present

| File | Size GB | Notes |
|------|--------:|-------|
| `data/graphs/roadNet-CA.tsv` | 0.077 | real SNAP graph |
| `data/graphs/com-youtube.tsv` | 0.036 | real SNAP graph, fallback URL supported by downloader |
| `data/graphs/wiki-talk.tsv` | 0.057 | real SNAP graph |
| `data/graphs/amazon0601.tsv` | 0.041 | real SNAP graph |
| `data/graphs/soc-livejournal.tsv` | 0.942 | real SNAP graph |

Downloaded graph TSVs are ignored by git; benchmark evidence is committed under `artifacts/`.

## Commands Run

```powershell
.\.venv\Scripts\python.exe scripts\download_graphs.py
.\.venv\Scripts\python.exe -m py_compile src\data_loader.py src\pagerank_cpu.py src\cpu_baseline.py src\benchmark.py src\gpu\pagerank_v1.py src\gpu\pagerank_v2.py src\gpu\pagerank_v3.py scripts\download_graphs.py
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe src\cpu_baseline.py --graph data\graphs\roadNet-CA.tsv
.\.venv\Scripts\python.exe src\benchmark.py
Select-String -Path artifacts\benchmark_results.csv -Pattern "com-youtube"
Select-String -Path src\gpu\*.py -Pattern "shfl_down_sync"
Select-String -Path src\gpu\pagerank_v1.py -Pattern "copy_to_host|to_device"
```

## Exact Benchmark Results

`python src\benchmark.py` completed successfully on the five required SNAP graphs and wrote `artifacts/benchmark_results.csv`.

| Graph | Best GPU | Time (s) | Iterations | Speedup vs CPU | Relative L1 vs SciPy | Spearman |
|-------|----------|---------:|-----------:|---------------:|---------------------:|---------:|
| roadNet-CA | gpu_v3_pull | 0.331933 | 57 | 20.389x | 2.175e-16 | 1.000 |
| com-youtube | gpu_v3_push | 0.065556 | 12 | 13.260x | 3.341e-13 | 1.000 |
| wiki-talk | gpu_v3_pull | 0.344802 | 40 | 14.805x | 2.605e-13 | 1.000 |
| amazon0601 | gpu_v3_push | 0.041988 | 55 | 65.887x | 3.227e-16 | 1.000 |
| soc-livejournal | gpu_v3_push | 1.171885 | 49 | 53.825x | 1.152e-14 | 1.000 |

Performance target: `com-youtube` converged in `0.065556s`, so the <= 5s target is met.

## Final Checklist

| Item | Command | Status |
|------|---------|--------|
| 1 | `.\.venv\Scripts\python.exe src\cpu_baseline.py --graph data\graphs\roadNet-CA.tsv` | PASS |
| 2 | `.\.venv\Scripts\python.exe -m pytest tests/ -v` | PASS |
| 3 | `.\.venv\Scripts\python.exe src\benchmark.py` | PASS |
| 4 | `Select-String -Path artifacts\benchmark_results.csv -Pattern "com-youtube"` | PASS |
| 5 | `Select-String -Path src\gpu\*.py -Pattern "shfl_down_sync"` | PASS |
| 6 | V1 transfer inspection with `Select-String` plus manual loop review | PASS |

## Remaining Issues

None known. Final verification: COMPLETE 10/10.

## Dashboard UI Session - 2026-05-29

Files created or modified:

- `scripts/run_dashboard.py`: FastAPI/uvicorn entry point for `python scripts/run_dashboard.py`, with optional `--port`.
- `src/ui/dashboard_server.py`: pure CSV parsing, graph detection, source checks, project status metadata, and API route setup.
- `src/ui/templates/index.html`: offline dashboard shell with eight sections and presentation mode.
- `src/ui/static/styles.css`: responsive light/dark UI styling with local system fonts and no CDN.
- `src/ui/static/app.js`: vanilla JavaScript for API loading, navigation, benchmark rendering, checklist rendering, and presentation mode.
- `tests/test_dashboard.py`: pure backend tests for CSV parsing, graph detection, and status shape.
- `docs/ui_visualization_plan.md`: dashboard purpose, tech stack decision, file tree, API contract, section inventory, limitations.
- `README.md`: appended Dashboard section.

Dashboard run command:

```powershell
python scripts/run_dashboard.py
```

Open `http://127.0.0.1:8000`.

Current benchmark data status: `artifacts/benchmark_results.csv` is present with 25 data rows. Schema is per-version: `graph_name`, `version`, `n_nodes`, `n_edges`, `convergence_time_s`, `iterations`, `iter_per_sec`, `cpu_spmv_total_s`, `version_time_over_cpu`, `speedup_vs_cpu`, `relative_l1_vs_scipy`, `spearman_vs_scipy`, `target_under_5s`, `cuda_available`, `note`.

Detected graph files: none in this workspace because `data/graphs/` is currently missing.

Open issues and limitations:

- Dashboard is visualization only and does not run `src/benchmark.py`.
- Pytest pass status and V1 host-copy loop review remain manual checklist items.
- The source check searches local `.cu`, `.cuh`, and Python files for `shfl_down_sync` because this project uses Numba CUDA Python source.
