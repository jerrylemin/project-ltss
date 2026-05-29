# LTSS Project Proposal - C3 PageRank

## Title + Team

- Title: PageRank
- Team:
  - Le Minh, Team Leader, integration, docs, GitHub.
  - Nguyen Vu Bach, GPU kernels, benchmark, testing.
- Git URL: https://github.com/jerrylemin/project-ltss
- Topic: C3 PageRank
- Track: Graph and Sparse

## Problem Statement

PageRank ranks nodes in a directed graph by repeatedly propagating rank mass along outgoing edges. The central workload is sparse matrix-vector multiplication over graph edges, making it suitable for GPU acceleration while retaining CPU orchestration.

## Dataset Details

The benchmark mode uses the five required SNAP targets: `roadNet-CA`, `com-youtube`, `wiki-talk`, `amazon0601`, and `soc-livejournal`. The downloader normalizes them to `data/graphs/*.tsv`; synthetic graphs remain available for smoke tests.

## Measured CPU Baseline Timing

Measured from the final five-graph benchmark in `artifacts/benchmark_results.csv`:

| Graph | CPU time (s) | Iterations | CPU mode |
|-------|-------------:|-----------:|----------|
| roadNet-CA | 6.767666 | 57 | custom CSR NumPy |
| com-youtube | 0.869269 | 12 | custom CSR NumPy |
| wiki-talk | 5.104843 | 40 | custom CSR NumPy |
| amazon0601 | 2.766479 | 55 | custom CSR NumPy |
| soc-livejournal | 63.076968 | 49 | custom CSR NumPy |

## Performance Target

PageRank on com-Youtube < 5s; 15-60x over CPU SpMV step where graph structure allows the target range. Final result: `gpu_v3_push` converged `com-youtube` in `0.065556s`.

## Bottleneck Analysis

Profile results in `artifacts/profile_summary.json` show `run_pagerank_cpu` and `_pagerank_numpy_loop` dominate runtime. The dominant algorithmic bottleneck is the SpMV/PageRank iteration: every iteration scans incoming CSR edges and gathers `rank[src] / out_degree[src]`.

## Optimization Plan

| Version | Strategy | Target Speedup |
|---------|----------|---------------|
| CPU Reference | NumPy CSR loop + SciPy verify | baseline |
| GPU V1 | CSR SpMV + damping kernels | 5-15x |
| GPU V2 | Fused SpMV+damping+L1 reduction | 15-40x |
| GPU V3 | Push vs Pull comparison, warp reduction | 30-60x |

## Final GPU Evidence

CUDA is usable on the current demo machine with `numba-cuda==0.30.2`. V3 pull and push keep rank vectors on device and copy only scalar reductions during iterations. GPU benchmark timings exclude warmup/JIT.

| Graph | Best GPU | GPU time (s) | Speedup | Relative L1 vs SciPy |
|-------|----------|-------------:|--------:|---------------------:|
| roadNet-CA | gpu_v3_pull | 0.331933 | 20.389x | 2.175e-16 |
| com-youtube | gpu_v3_push | 0.065556 | 13.260x | 3.341e-13 |
| wiki-talk | gpu_v3_pull | 0.344802 | 14.805x | 2.605e-13 |
| amazon0601 | gpu_v3_push | 0.041988 | 65.887x | 3.227e-16 |
| soc-livejournal | gpu_v3_push | 1.171885 | 53.825x | 1.152e-14 |

The complete all-version result table is in `artifacts/benchmark_results.csv`.

## Risk Analysis

- GPU unavailable or CUDA runtime incompatible with Numba. Mitigation: CPU baseline remains fully functional and GPU tests skip cleanly when CUDA is unavailable.
- SNAP datasets are large and should not be committed. Mitigation: `data/graphs/*` is ignored and benchmark evidence is committed as CSV/JSON.
- Atomic floating-point precision can affect push-scatter reproducibility. Mitigation: CPU and SciPy references verify rank sums and relative errors; final GPU relative errors are below `1e-6`.
- Numba CUDA limitations may require simpler kernels than hand-written CUDA C++. Mitigation: V3 uses `cuda.shfl_down_sync` for warp-level L1 reduction and keeps a stable CPU/SciPy reference for validation.

## Division of Work

- Le Minh: integration, docs, GitHub, proposal, environment handoff.
- Nguyen Vu Bach: GPU kernels, benchmark harness, testing and correctness checks.
