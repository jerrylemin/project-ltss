# Codex Context

Last updated: 2026-05-29T14:03:47+07:00

This repository implements C3 PageRank for the Applied Parallel Programming Graph and Sparse track. The CPU path uses an incoming CSR representation for PageRank pull/gather timing and a SciPy CSR reference for verification. GPU code is optional and isolated under `src/gpu/`; CUDA unavailable paths should skip cleanly.

Current implementation notes:

- `src/data_loader.py` now builds both incoming CSR (`indptr`, `indices`) and outgoing CSR (`indptr_out`, `indices_out`).
- `src/gpu/pagerank_v3.py` has V3 pull and source-owned V3 push kernels with explicit `cuda.shfl_down_sync` warp-level L1 reductions.
- `scripts/download_graphs.py` downloads the five required SNAP graphs into ignored local files under `data/graphs/`.
- `src/gpu/pagerank_v2.py` uses a fused SpMV+damping kernel with block-level shared-memory L1 reduction.
- `src/benchmark.py` defaults to the five SNAP graph paths in `BENCHMARK_GRAPHS` and saves the course table to `artifacts/benchmark_results.csv`.
- Full real-data benchmark completed with CUDA on NVIDIA GeForce RTX 3060 Laptop GPU. `com-youtube` target is met by `gpu_v3_push` at `0.065556s`, relative L1 error `3.341e-13` vs SciPy.

Do not commit downloaded SNAP graph data. Use benchmark CSV/JSON summaries as committed evidence instead.
