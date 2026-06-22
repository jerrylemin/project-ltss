# Benchmark Report

Last updated: 2026-06-13

## Environment

- Workspace: `C:\Users\Administrator\Documents\MEGA\project-ltss`
- Python: `3.11.9` in `.venv`
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Driver: `591.86`
- GPU memory: `6144 MiB`
- CUDA available through Numba: `True`
- Package pins: `numpy==2.4.6`, `scipy==1.17.1`, `numba==0.65.1`, `numba-cuda==0.30.2`
- Tolerance: `1e-6`
- Max iterations: `200`
- Correctness reference: SciPy sparse PageRank

## Dataset Table

| Graph | Nodes | Edges | Notes |
|---|---:|---:|---|
| roadNet-CA | 1,971,281 | 5,533,214 | Directed road network |
| com-youtube | 1,157,828 | 2,987,624 | SNAP YouTube graph |
| wiki-talk | 2,394,385 | 5,021,410 | Wikipedia talk graph |
| amazon0601 | 403,394 | 3,387,388 | Amazon product graph |
| soc-livejournal | 4,847,571 | 68,993,773 | Self-loops preserved |

## Timing Methodology

Final benchmark command:

```powershell
.\.venv\Scripts\python.exe src\benchmark.py --graphs all --repeat 5 --warmup 2 --output artifacts\benchmark_results.csv --include-transfer-timing
```

Method:

- Five timed repeats per graph/version.
- Two untimed CUDA warmup runs per GPU graph/version.
- Warmup excludes first-call Numba JIT compilation from timed medians.
- GPU runners call `cuda.synchronize()` after H2D setup and after timed kernel loops.
- Transfer timing boundaries are reported as H2D, kernel/convergence, D2H, total, and repeat statistics.
- CPU and GPU use the same graph structure, tolerance, max iteration count, dangling handling, self-loop handling, and SciPy reference.

Self-loops and duplicate edges are preserved. This matters for `soc-livejournal`, whose edge count is now the full local TSV count rather than the old self-loop-filtered count.

## Real-Graph CPU Profile

Command:

```powershell
.\.venv\Scripts\python.exe src\cpu_baseline.py --graph amazon0601 --profile --output artifacts\profile_summary.json
```

| Graph | Iterations | SpMV time (s) | Damping time (s) | L1 convergence time (s) | Total compute time (s) | SpMV percentage |
|---|---:|---:|---:|---:|---:|---:|
| amazon0601 | 55 | 2.573834 | 0.084451 | 0.096738 | 2.816803 | 91.37% |

The measured bottleneck is SpMV: repeated CSR edge traversal dominates PageRank compute time on a real SNAP graph.

## Final Repeat Benchmark

`convergence_time_median_s` is the median over five timed repeats. Speedup is computed against CPU SpMV median for the same graph.

| Graph | Version | Iterations | Median time (s) | Mean time (s) | Stddev (s) | CPU SpMV median (s) | Speedup vs CPU SpMV | Rel L1 vs SciPy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| roadNet-CA | gpu_v1 | 57 | 0.451783 | 0.461173 | 0.016207 | 5.339604 | 11.819x | 2.175e-16 |
| roadNet-CA | gpu_v2 | 57 | 0.347213 | 0.360096 | 0.030305 | 5.339604 | 15.378x | 2.175e-16 |
| roadNet-CA | gpu_v3_pull | 57 | 0.119364 | 0.120335 | 0.002705 | 5.339604 | 44.734x | 2.175e-16 |
| roadNet-CA | gpu_v3_push | 57 | 0.062436 | 0.062722 | 0.000640 | 5.339604 | 85.521x | 1.533e-16 |
| com-youtube | gpu_v1 | 12 | 0.274649 | 0.285768 | 0.027483 | 0.647530 | 2.358x | 3.328e-13 |
| com-youtube | gpu_v2 | 12 | 0.292664 | 0.310961 | 0.039412 | 0.647530 | 2.213x | 3.342e-13 |
| com-youtube | gpu_v3_pull | 12 | 0.069220 | 0.069190 | 0.000111 | 0.647530 | 9.355x | 3.287e-13 |
| com-youtube | gpu_v3_push | 12 | 0.052838 | 0.052854 | 0.000123 | 0.647530 | 12.255x | 3.291e-13 |
| wiki-talk | gpu_v1 | 40 | 0.637475 | 0.637290 | 0.003547 | 3.656651 | 5.736x | 2.606e-13 |
| wiki-talk | gpu_v2 | 40 | 0.542198 | 0.565505 | 0.047348 | 3.656651 | 6.744x | 2.591e-13 |
| wiki-talk | gpu_v3_pull | 40 | 0.320790 | 0.320943 | 0.000275 | 3.656651 | 11.399x | 2.601e-13 |
| wiki-talk | gpu_v3_push | 40 | 0.530892 | 0.530843 | 0.000485 | 3.656651 | 6.888x | 2.601e-13 |
| amazon0601 | gpu_v1 | 55 | 0.380197 | 0.380081 | 0.001237 | 2.449847 | 6.444x | 1.665e-16 |
| amazon0601 | gpu_v2 | 55 | 0.389405 | 0.415943 | 0.053939 | 2.449847 | 6.291x | 1.664e-16 |
| amazon0601 | gpu_v3_pull | 55 | 0.161648 | 0.161573 | 0.001269 | 2.449847 | 15.155x | 1.667e-16 |
| amazon0601 | gpu_v3_push | 55 | 0.034461 | 0.034780 | 0.000695 | 2.449847 | 71.090x | 3.214e-16 |
| soc-livejournal | gpu_v1 | 51 | 2.250822 | 2.248321 | 0.009211 | 62.821708 | 27.911x | 7.682e-15 |
| soc-livejournal | gpu_v2 | 51 | 1.992230 | 2.030075 | 0.077563 | 62.821708 | 31.533x | 7.658e-15 |
| soc-livejournal | gpu_v3_pull | 51 | 1.783850 | 1.789119 | 0.010083 | 62.821708 | 35.217x | 7.807e-15 |
| soc-livejournal | gpu_v3_push | 51 | 1.141057 | 1.141025 | 0.001120 | 62.821708 | 55.056x | 7.910e-15 |

## com-youtube Target

Best repeat-median result: `gpu_v3_push`, `0.052838s`, relative L1 error around `3.29e-13` vs SciPy.

Status: `TARGET MET`.

## Push Versus Pull

| Graph | V3 pull median (s) | V3 push median (s) | Winner |
|---|---:|---:|---|
| roadNet-CA | 0.119364 | 0.062436 | push |
| com-youtube | 0.069220 | 0.052838 | push |
| wiki-talk | 0.320790 | 0.530892 | pull |
| amazon0601 | 0.161648 | 0.034461 | push |
| soc-livejournal | 1.783850 | 1.141057 | push |

Pull gathers incoming contributions and avoids atomic scatter. Push traverses outgoing CSR and uses atomic adds; it wins when outgoing traversal locality and degree distribution outweigh atomic contention. Degree skew and high-degree rows explain why winners vary by graph.

## Limitations

- The benchmark does not collect occupancy or memory-bandwidth counters.
- Windows WDDM scheduling and GPU thermals can shift timings.
- Large raw SNAP graph files remain local and ignored by git.
