# Project Structure

Last updated: 2026-05-29

```text
project-ltss/
├── README.md
├── requirements.txt
├── project_spec.yaml
├── src/
│   ├── cpu_baseline.py
│   ├── pagerank_cpu.py
│   ├── data_loader.py
│   ├── benchmark.py
│   └── gpu/
│       ├── pagerank_v1.py
│       ├── pagerank_v2.py
│       └── pagerank_v3.py
├── tests/
├── scripts/
├── docs/
├── artifacts/
└── data/
```

Key ownership boundaries:

- CPU correctness and timing live in `src/pagerank_cpu.py`.
- Graph construction and SNAP-style edge-list parsing live in `src/data_loader.py`.
- Optional CUDA implementations live under `src/gpu/`.
- Benchmark orchestration lives in `src/benchmark.py`.
- Large local datasets belong under `data/raw/` or `data/graphs/` and are ignored by git.
