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
- Elapsed seconds: 0.13196079999943322
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

## Final GPU Evidence

CUDA is usable on the current demo machine after installing CUDA Toolkit 13.2 and `numba-cuda==0.30.2`. V3 pull was optimized to keep rank vectors on device and copy only scalar reductions during iterations. GPU benchmark timings exclude warmup/JIT.

| Graph | CPU time | V3 pull time | Speedup | Relative error |
|-------|---------:|-------------:|--------:|---------------:|
| roadNet-CA-sample200k | 7.697041500001433 | 0.01401340000120399 | 549.262955409831x | 6.526240627828615e-16 |
| synthetic_large | 7.975994500000525 | 0.01300710000032268 | 613.2031351955975x | 4.59684568716672e-16 |

Full roadNet-CA was downloaded locally but not committed. The measured SNAP evidence uses a 200k-edge roadNet-CA sample to keep CPU baseline runtime reasonable for demo.

## Risk Analysis

- GPU unavailable or CUDA runtime incompatible with Numba.
- SNAP datasets may be too large for local disk, RAM, or submission workflow.
- Atomic floating-point precision can affect push-scatter reproducibility.
- Numba CUDA limitations may require simpler kernels than hand-written CUDA C++.

## Division of Work

- Le Minh: integration, docs, GitHub, proposal, environment handoff.
- Nguyen Vu Bach: GPU kernels, benchmark harness, testing and correctness checks.
