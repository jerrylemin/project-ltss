from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "final_report.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        markdown(
            """
# GPU-Accelerated PageRank on Large Graphs

Applied Parallel Programming, Track C3 PageRank. This notebook is the executable final report for the LTSS PageRank project.
"""
        ),
        markdown(
            """
## Project overview

The project implements PageRank for directed sparse graphs with a CPU SciPy reference, an own NumPy CSR timing baseline, and three Numba CUDA implementations. The target recurrence is `r_new = alpha * A * r + (1 - alpha) * e`, with dangling mass redistribution, L1 convergence, rank normalization, preserved self-loops, and preserved duplicate edges.
"""
        ),
        markdown(
            """
## C3 requirement mapping

| Requirement | Evidence |
|---|---|
| CPU SciPy reference | `src/pagerank_cpu.py` |
| Own CPU CSR timing baseline | `src/pagerank_cpu.py`, `src/cpu_baseline.py` |
| GPU V1 CSR SpMV | `src/gpu/pagerank_v1.py` |
| GPU V2 fused SpMV, damping, L1 | `src/gpu/pagerank_v2.py` |
| GPU V3 pull vs push | `src/gpu/pagerank_v3.py` |
| Warp shuffle reduction | `cuda.shfl_down_sync` in V3 |
| Five real SNAP graphs | `artifacts/benchmark_results.csv` |
| Correctness within `1e-6` | SciPy relative L1 columns and tests |
"""
        ),
        markdown(
            """
## PageRank formula

The implementation uses incoming CSR for pull/gather: each destination row accumulates `rank[src] / out_degree[src]`. Dangling nodes contribute their rank mass uniformly to every node. Every iteration normalizes the rank vector and stops when the L1 delta is below `1e-6`.
"""
        ),
        code(
            """
from pathlib import Path
import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
BENCHMARK_CSV = ARTIFACTS / "benchmark_results.csv"
PROFILE_JSON = ARTIFACTS / "profile_summary.json"

pd.set_option("display.max_columns", 80)
print("Workspace:", ROOT)
print("Benchmark artifact exists:", BENCHMARK_CSV.exists())
print("Profile artifact exists:", PROFILE_JSON.exists())
"""
        ),
        markdown(
            """
## Dataset table

The notebook reads existing artifacts and does not download the large SNAP files during execution.
"""
        ),
        code(
            """
if BENCHMARK_CSV.exists():
    benchmark = pd.read_csv(BENCHMARK_CSV)
else:
    smoke_path = ARTIFACTS / "notebook_smoke_benchmark.csv"
    subprocess.run(
        [sys.executable, "src/benchmark.py", "--synthetic", "--sizes", "small", "--versions", "cpu", "--repeat", "1", "--output", str(smoke_path), "--no-write-default-artifacts"],
        check=True,
    )
    benchmark = pd.read_csv(smoke_path)

time_col = "convergence_time_median_s" if "convergence_time_median_s" in benchmark.columns else "convergence_time_s"
dataset_table = (
    benchmark[["graph_name", "n_nodes", "n_edges"]]
    .dropna()
    .drop_duplicates()
    .sort_values("graph_name")
    .reset_index(drop=True)
)
dataset_table
"""
        ),
        markdown(
            """
## CPU baseline design

The CPU baseline uses the same incoming CSR graph as the GPU pull implementations. It times SpMV, dangling mass, damping, normalization, and L1 convergence separately so the bottleneck claim is measured rather than assumed.
"""
        ),
        markdown(
            """
## Profiling evidence
"""
        ),
        code(
            """
if PROFILE_JSON.exists():
    profile = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
else:
    profile = {}

profile_rows = pd.DataFrame(
    [
        {"component": "dangling mass", "seconds": profile.get("dangling_mass_total_seconds")},
        {"component": "SpMV", "seconds": profile.get("spmv_total_seconds")},
        {"component": "damping", "seconds": profile.get("damping_total_seconds")},
        {"component": "normalization", "seconds": profile.get("normalization_total_seconds")},
        {"component": "L1 convergence", "seconds": profile.get("convergence_l1_total_seconds")},
    ]
).dropna()
print("Profile graph:", profile.get("graph_name"))
print("SpMV percent of iteration time:", profile.get("spmv_percent_iteration_time"))
profile_rows
"""
        ),
        markdown(
            """
## Bottleneck justification

SpMV is the repeated edge traversal step. On the profiled real graph, the measured SpMV percentage is computed from per-iteration timers and dominates the total PageRank compute time.
"""
        ),
        markdown(
            """
## GPU V1 design

V1 assigns one CUDA thread per destination row and gathers incoming CSR contributions. Damping and convergence are handled in a separate update kernel.

## GPU V2 design

V2 fuses CSR SpMV, damping update, and L1 convergence accumulation into one row-parallel kernel with shared-memory block reduction.

## GPU V3 pull design

V3 pull keeps the gather formulation but uses warp-level shuffle reduction for convergence.

## GPU V3 push design

V3 push assigns one thread per source node and scatters contributions through outgoing CSR with `cuda.atomic.add`.

## Warp shuffle reduction

V3 uses `cuda.shfl_down_sync` to reduce per-thread L1 deltas within a warp before one atomic add per warp.
"""
        ),
        markdown(
            """
## Dangling node handling

Dangling mass is the sum of ranks for nodes with zero out-degree. The implementation adds `alpha * dangling_mass / n` to every destination every iteration.

## Self-loop handling

Self-loops are preserved. A self-loop `u -> u` contributes to `u` exactly like any other outgoing edge and counts in `out_degree[u]`. Duplicate edges are also preserved consistently across CPU, SciPy, and GPU paths.
"""
        ),
        markdown(
            """
## Correctness methodology

All implementations compare against SciPy sparse PageRank with relative L1 error threshold `1e-6`. This notebook also executes a small self-loop regression demo.
"""
        ),
        code(
            """
from src.data_loader import _graph_from_edges
from src.pagerank_cpu import run_pagerank_cpu
from src.metrics import relative_l1_error

edges = [(0, 0), (0, 1), (1, 2), (2, 0)]
graph = _graph_from_edges(edges, num_nodes=3, name="notebook_self_loop", remap=False)
rank_cpu, _ = run_pagerank_cpu(graph, verify_scipy=False)

src = np.array([s for s, _ in edges], dtype=np.int64)
dst = np.array([d for _, d in edges], dtype=np.int64)
out_degree = np.bincount(src, minlength=3)
matrix = sp.csr_matrix((1.0 / out_degree[src], (dst, src)), shape=(3, 3))
rank = np.full(3, 1 / 3, dtype=np.float64)
alpha = 0.85
for _ in range(200):
    new_rank = (1 - alpha) / 3 + alpha * (matrix @ rank + rank[out_degree == 0].sum() / 3)
    new_rank /= new_rank.sum()
    if np.abs(new_rank - rank).sum() < 1e-6:
        break
    rank = np.asarray(new_rank)

demo_error = relative_l1_error(rank_cpu, rank)
assert graph["num_edges"] == 4
assert demo_error <= 1e-6
pd.DataFrame([{"case": "self-loop demo", "edges": graph["num_edges"], "relative_l1_error": demo_error, "status": "PASS"}])
"""
        ),
        markdown(
            """
## Correctness results
"""
        ),
        code(
            """
correctness_cols = ["graph_name", "version", "relative_l1_vs_scipy", "spearman_vs_scipy"]
available = [c for c in correctness_cols if c in benchmark.columns]
benchmark[available].dropna(subset=["relative_l1_vs_scipy"]).head(25)
"""
        ),
        markdown(
            """
## Benchmark methodology

Final benchmarks use CUDA warmup runs to exclude JIT compilation, repeated timed runs, synchronized GPU timings, and median/mean/min/max/stddev statistics. CPU and GPU use the same graph, same tolerance, and same maximum iteration count.
"""
        ),
        markdown(
            """
## Five-graph benchmark table
"""
        ),
        code(
            """
summary_cols = [
    "graph_name", "version", "n_nodes", "n_edges", "iterations", time_col,
    "convergence_time_mean_s", "convergence_time_std_s", "speedup_vs_cpu_spmv",
    "relative_l1_vs_scipy",
]
summary_cols = [c for c in summary_cols if c in benchmark.columns]
benchmark[summary_cols].head(30)
"""
        ),
        markdown(
            """
## Speedup chart
"""
        ),
        code(
            """
plot_df = benchmark.copy()
plot_df["time_value"] = pd.to_numeric(plot_df[time_col], errors="coerce")
if "cpu_spmv_median_s" in plot_df.columns:
    plot_df["cpu_spmv_median_s"] = pd.to_numeric(plot_df["cpu_spmv_median_s"], errors="coerce")
    plot_df["speedup_plot"] = plot_df["cpu_spmv_median_s"] / plot_df["time_value"]
else:
    plot_df["speedup_plot"] = pd.to_numeric(plot_df.get("speedup_vs_cpu", np.nan), errors="coerce")
gpu_plot = plot_df[plot_df["version"].astype(str).str.startswith("gpu_")].dropna(subset=["speedup_plot"])
if not gpu_plot.empty:
    ax = gpu_plot.pivot_table(index="graph_name", columns="version", values="speedup_plot", aggfunc="max").plot(kind="bar", figsize=(10, 4))
    ax.set_ylabel("Speedup vs CPU SpMV")
    ax.set_title("GPU PageRank Speedup by Graph")
    plt.tight_layout()
else:
    print("No GPU speedup rows available in artifact.")
"""
        ),
        markdown(
            """
## Push versus pull discussion

Pull avoids atomic scatter and is often stronger when incoming row traversal is balanced. Push can win when outgoing traversal locality and graph structure offset atomic contention.

## Load imbalance discussion

Sparse web and social graphs have irregular degree distributions. Row-parallel pull paths can underutilize threads on short rows and serialize work on high-degree rows. Push paths can concentrate atomic updates on hubs.

## Limitations

Timing depends on GPU thermals and Windows WDDM scheduling. The benchmark reports transfer and kernel timing boundaries but does not include low-level occupancy counters.
"""
        ),
        markdown(
            """
## Reproducible commands

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip setuptools wheel
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pytest tests/ -v
.\\.venv\\Scripts\\python.exe src\\cpu_baseline.py --graph amazon0601 --profile --output artifacts\\profile_summary.json
.\\.venv\\Scripts\\python.exe src\\benchmark.py --graphs all --repeat 5 --warmup 2 --output artifacts\\benchmark_results.csv --include-transfer-timing
.\\.venv\\Scripts\\jupyter.exe nbconvert --to notebook --execute notebooks\\final_report.ipynb --output executed_report.ipynb
```
"""
        ),
        markdown(
            """
## Final conclusion

The project satisfies C3 PageRank with real CPU and CUDA implementations, SciPy correctness validation, preserved self-loop and dangling-node semantics, five real SNAP graph benchmarks, and an executable final report.
"""
        ),
    ]

    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"final-report-{index:02d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
