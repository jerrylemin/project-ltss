from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import algorithm_params, load_config
from src.data_loader import load_graph_from_config
from src.gpu.cuda_utils import cuda_available, cuda_status
from src.gpu.pagerank_gpu import run_pagerank_gpu
from src.metrics import relative_l1_error, write_json
from src.pagerank_cpu import run_pagerank_cpu


COLUMNS = [
    "backend",
    "version",
    "graph_name",
    "num_nodes",
    "num_edges",
    "iterations",
    "elapsed_seconds",
    "iterations_per_second",
    "speedup_vs_cpu",
    "relative_error_vs_cpu",
    "cuda_available",
    "note",
]


def _iterations_per_second(iterations: int | None, elapsed: float | None) -> float | None:
    if not elapsed:
        return None
    return float(iterations or 0) / float(elapsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU and optional GPU PageRank.")
    parser.add_argument("--config", default="project_spec.yaml")
    parser.add_argument("--graph", default="small")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    alpha, tol, max_iter = algorithm_params(config)
    graph = load_graph_from_config(config, graph_size=args.graph)

    cpu_rank, cpu_metrics = run_pagerank_cpu(graph, alpha=alpha, tol=tol, max_iter=max_iter)
    cpu_elapsed = float(cpu_metrics["elapsed_seconds"])
    rows: list[dict[str, Any]] = [
        {
            "backend": "cpu",
            "version": cpu_metrics["mode"],
            "graph_name": graph["name"],
            "num_nodes": graph["num_nodes"],
            "num_edges": graph["num_edges"],
            "iterations": cpu_metrics["iterations"],
            "elapsed_seconds": cpu_elapsed,
            "iterations_per_second": _iterations_per_second(cpu_metrics["iterations"], cpu_elapsed),
            "speedup_vs_cpu": 1.0,
            "relative_error_vs_cpu": 0.0,
            "cuda_available": cuda_available(),
            "note": "",
        }
    ]

    for version in ("v1", "v2", "v3_pull", "v3_push"):
        if not cuda_available():
            rows.append(
                {
                    "backend": "gpu",
                    "version": version,
                    "graph_name": graph["name"],
                    "num_nodes": graph["num_nodes"],
                    "num_edges": graph["num_edges"],
                    "iterations": None,
                    "elapsed_seconds": None,
                    "iterations_per_second": None,
                    "speedup_vs_cpu": None,
                    "relative_error_vs_cpu": None,
                    "cuda_available": False,
                    "note": "skipped: no cuda",
                }
            )
            continue
        try:
            gpu_rank, gpu_metrics = run_pagerank_gpu(graph, alpha=alpha, tol=tol, max_iter=max_iter, version=version)
            elapsed = float(gpu_metrics["elapsed_seconds"])
            rows.append(
                {
                    "backend": "gpu",
                    "version": version,
                    "graph_name": graph["name"],
                    "num_nodes": graph["num_nodes"],
                    "num_edges": graph["num_edges"],
                    "iterations": gpu_metrics["iterations"],
                    "elapsed_seconds": elapsed,
                    "iterations_per_second": _iterations_per_second(gpu_metrics["iterations"], elapsed),
                    "speedup_vs_cpu": cpu_elapsed / elapsed if elapsed else None,
                    "relative_error_vs_cpu": relative_l1_error(gpu_rank, cpu_rank),
                    "cuda_available": True,
                    "note": gpu_metrics.get("fallback_note", ""),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "backend": "gpu",
                    "version": version,
                    "graph_name": graph["name"],
                    "num_nodes": graph["num_nodes"],
                    "num_edges": graph["num_edges"],
                    "iterations": None,
                    "elapsed_seconds": None,
                    "iterations_per_second": None,
                    "speedup_vs_cpu": None,
                    "relative_error_vs_cpu": None,
                    "cuda_available": cuda_available(),
                    "note": f"skipped: {exc}",
                }
            )

    Path("artifacts").mkdir(exist_ok=True)
    with Path("artifacts/benchmarks.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        "artifacts/benchmark_summary.json",
        {
            "graph_name": graph["name"],
            "num_nodes": graph["num_nodes"],
            "num_edges": graph["num_edges"],
            "rows": rows,
            "cuda_status": cuda_status(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
