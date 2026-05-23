# Project Structure

- `src/data_loader.py`: SNAP-style edge list parser and synthetic graph generator.
- `src/pagerank_cpu.py`: NumPy CSR-loop PageRank baseline plus optional SciPy verification.
- `src/cpu_baseline.py`: CLI for baseline metrics and `artifacts/cpu_baseline_metrics.json`.
- `src/profile_cpu.py`: cProfile harness and bottleneck document generation.
- `src/gpu/`: optional Numba CUDA PageRank versions.
- `src/benchmark.py`: CPU/GPU benchmark CSV and summary JSON generation.
- `tests/`: correctness, loader, config, CLI, and optional GPU tests.
- `docs/`: proposal, benchmark, setup, handoff, and submission notes.
- `scripts/`: Windows PowerShell helper commands.
