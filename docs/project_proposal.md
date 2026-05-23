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

The default dataset mode is synthetic so the project runs without external downloads. Optional SNAP targets are `roadNet-CA`, `com-Youtube`, `wiki-Talk`, `amazon0601`, and `soc-LiveJournal1` when local edge-list files are available.

## Measured CPU Baseline Timing

Measured from `artifacts/cpu_baseline_metrics.json` on synthetic small graph:

- Nodes: 1000
- Edges: 5000
- Iterations: 16
- Elapsed seconds: 0.109798999999839
- Final L1 delta: 6.046211883552536e-07
- Rank sum: 1.0
- CPU mode: NumPy CSR loop

## Performance Target

PageRank on com-Youtube < 5s; 15-60x over CPU SpMV step.

## Bottleneck Analysis

Profile results in `artifacts/profile_summary.json` show `run_pagerank_cpu` and `_pagerank_numpy_loop` dominate runtime. The dominant algorithmic bottleneck is the SpMV/PageRank iteration: every iteration scans incoming CSR edges and gathers `rank[src] / out_degree[src]`.

## Optimization Plan

| Version | Strategy | Target Speedup |
|---------|----------|---------------|
| CPU Reference | NumPy CSR loop + SciPy verify | baseline |
| GPU V1 | CSR SpMV + damping kernels | 5-15x |
| GPU V2 | Fused SpMV+damping+L1 reduction | 15-40x |
| GPU V3 | Push vs Pull comparison, warp reduction | 30-60x |

## Preliminary GPU Result

CUDA is usable on the current demo machine after installing CUDA Toolkit 13.2 and `numba-cuda==0.30.2`. GPU benchmark rows are generated for V1, V2, V3 pull, and V3 push on the synthetic small graph. The graph is intentionally small for CI/demo, so GPU speedups are below CPU because kernel launch and JIT overhead dominate; relative error versus CPU remains around 1e-16.

## Risk Analysis

- GPU unavailable or CUDA runtime incompatible with Numba.
- SNAP datasets may be too large for local disk, RAM, or submission workflow.
- Atomic floating-point precision can affect push-scatter reproducibility.
- Numba CUDA limitations may require simpler kernels than hand-written CUDA C++.

## Division of Work

- Le Minh: integration, docs, GitHub, proposal, environment handoff.
- Nguyen Vu Bach: GPU kernels, benchmark harness, testing and correctness checks.
