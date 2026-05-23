# Codex Context

Project C3 PageRank for LTSS, track Graph and Sparse.

Default execution uses synthetic graphs so the CPU baseline, tests, profiling, and benchmark harness work without external datasets.

The GPU implementation is optional and auto-detected through Numba CUDA. CPU code is the reliable baseline and must pass on machines without CUDA.
