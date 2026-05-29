# Feature Progress

Last updated: 2026-05-29T14:03:47+07:00

## Completed

- Offline FastAPI dashboard for PageRank CUDA demo visualization, including API endpoints, local static UI, benchmark CSV rendering, graph-file status, checklist, and presentation mode.
- CPU PageRank baseline with custom CSR timing loop and SciPy reference.
- Optional CUDA V1/V2/V3 entry points.
- Incoming and outgoing CSR graph structures in the loader.
- V3 pull/gather and source-owned V3 push/scatter kernels with warp-shuffle L1 reduction.
- SNAP downloader script for the five required graphs.
- V2 fused kernel now uses a 256-thread shared-memory L1 reduction with one atomic add per block.
- GPU V1/V2/V3 keep rank vectors on device during iterations and only copy scalar convergence values plus final rank vectors.
- `src/benchmark.py` now defines the five required SNAP graphs and writes `artifacts/benchmark_results.csv` by default.
- `tests/test_correctness.py` now covers CPU-vs-SciPy at `1e-6`, dangling-node CPU behavior, GPU V1-vs-SciPy, V2-vs-V1, V3 push-vs-pull, and V3-vs-SciPy with CUDA skip markers.
- `requirements.txt` is generated from the project environment with exact `==` pins.
- CPU baseline metrics include SpMV-only per-iteration timings and printed SpMV summary lines.
- Benchmark output prints V2/V1, V3/V2, and best-GPU-vs-CPU-SpMV speedup ratios when the required timing rows are present.
- README now includes final five-SNAP-graph results, CUDA/setup/download/final verification commands, and measured push vs pull performance analysis.
- Full real-data benchmark completed on roadNet-CA, com-youtube, wiki-talk, amazon0601, and soc-livejournal with CUDA execution on RTX 3060 Laptop GPU.
- `com-youtube` target is met: best GPU is `gpu_v3_push` at `0.065556s`, tolerance `1e-6`, relative L1 error `3.341e-13` vs SciPy.

## Pending

- Dashboard manual browser checklist should be run during the next live UI review if a browser is available.
