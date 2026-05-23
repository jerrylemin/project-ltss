# Feature Progress

## Completed

- Initial project spec and repo structure scaffold.
- CPU PageRank baseline with synthetic and edge-list graph loading.
- Optional GPU API and benchmark harness.
- Test suite for config, data loader, correctness, CLI, and GPU skip behavior.
- Profile and benchmark artifacts generated.
- Team member roles filled in proposal and README.
- CUDA Toolkit 13.2 plus `numba-cuda` validated; GPU smoke and benchmark scripts added.
- SNAP sample runner and dataset setup docs added.

## Pending

- Run with real SNAP datasets if local paths become available.
- Tune GPU kernels on larger SNAP graphs; current synthetic small graph is dominated by launch/JIT overhead.
