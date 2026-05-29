# Codex Context

Last updated: 2026-05-29T18:31:00+07:00

This repository implements C3 PageRank for the Applied Parallel Programming Graph and Sparse track. The CPU path uses incoming CSR for PageRank pull/gather timing and SciPy CSR as the correctness reference. GPU code is optional and isolated under `src/gpu/`; CUDA-unavailable paths should skip cleanly.

Current implementation notes:

- `src/data_loader.py` builds incoming CSR (`indptr`, `indices`) and outgoing CSR (`indptr_out`, `indices_out`) plus `out_degree`.
- `src/gpu/pagerank_v1.py` has separate `csr_spmv_kernel` and `damping_update_kernel`.
- `src/gpu/pagerank_v2.py` uses a fused SpMV+damping+L1 kernel with block-level shared-memory reduction.
- `src/gpu/pagerank_v3.py` has V3 pull and source-owned V3 push kernels with explicit `cuda.shfl_down_sync` warp-level L1 reductions.
- `scripts/download_graphs.py` downloads the five required SNAP graphs into ignored local files under `data/graphs/`.
- `src/benchmark.py` defaults to the five SNAP graph paths in `BENCHMARK_GRAPHS` and saves the course table to `artifacts/benchmark_results.csv`.
- `.venv` was rebuilt with Python 3.12.6. Python 3.10 cannot install the current pinned dependencies.
- Full real-data benchmark completed with CUDA on NVIDIA GeForce RTX 3060. `com-youtube` target is met by `gpu_v3_push` at `0.127168s`, relative L1 error `3.320e-13` vs SciPy.

Do not commit downloaded SNAP graph data. Use benchmark CSV/JSON summaries as committed evidence instead.
