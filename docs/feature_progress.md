# Feature Progress

Last updated: 2026-05-29T18:31:00+07:00

## Completed

- Clean `.venv` rebuilt with Python 3.12.6 after the previous Python 3.10 environment could not install the pinned Python >=3.11 packages.
- `requirements.txt` now includes exact pins for FastAPI, uvicorn, and their transitive runtime packages so a fresh setup can start the dashboard.
- CPU PageRank baseline with custom CSR timing loop and SciPy reference.
- Incoming and outgoing CSR graph structures in the loader.
- Optional CUDA V1/V2/V3 entry points, with V1 separate SpMV/damping, V2 fused kernel, and V3 pull/push modes.
- V3 pull/gather and source-owned V3 push/scatter kernels use `cuda.shfl_down_sync` warp-level reduction.
- SNAP downloader script downloaded all five required graphs under ignored `data/graphs/*.tsv`.
- `src/benchmark.py` produced a 25-row five-graph x five-version benchmark table at `artifacts/benchmark_results.csv`.
- `python -m pytest tests/ -v` passed: 21 passed, including GPU tests on CUDA.
- Dashboard endpoint smoke passed for `/`, `/api/benchmark`, `/api/graphs`, `/api/status`, and `/api/shfl_check`.
- `com-youtube` target is met: best GPU is `gpu_v3_push` at `0.127168s`, tolerance `1e-6`, relative L1 error `3.320e-13` vs SciPy.

## Pending

- No known submission blockers. Manual live presentation fit check can still be done in a browser before grading.
