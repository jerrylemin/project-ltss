from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import algorithm_params, load_config
from src.data_loader import load_edge_list, load_graph_from_config
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
    "warmup_excluded",
    "note",
]


def _iterations_per_second(iterations: int | None, elapsed: float | None) -> float | None:
    if not elapsed:
        return None
    return float(iterations or 0) / float(elapsed)


def _parse_csv_arg(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_graphs(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    if args.edges_path:
        graph = load_edge_list(args.edges_path)
        if args.graph_name:
            graph = {**graph, "name": args.graph_name}
        return [graph]

    graph_sizes = _parse_csv_arg(args.sizes, [args.graph])
    return [load_graph_from_config(config, graph_size=size) for size in graph_sizes]


def _benchmark_graph(
    graph: dict[str, Any],
    *,
    alpha: float,
    tol: float,
    max_iter: int,
    versions: list[str],
    run_gpu: bool,
    warmup_gpu: bool,
    verify_scipy: bool,
) -> list[dict[str, Any]]:
    cpu_rank, cpu_metrics = run_pagerank_cpu(
        graph, alpha=alpha, tol=tol, max_iter=max_iter, verify_scipy=verify_scipy
    )
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
            "warmup_excluded": False,
            "note": "",
        }
    ]

    for version in versions:
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
                    "warmup_excluded": warmup_gpu,
                    "note": "skipped: no cuda",
                }
            )
            continue
        try:
            warmup_note = ""
            if run_gpu and warmup_gpu:
                run_pagerank_gpu(graph, alpha=alpha, tol=tol, max_iter=1, version=version)
                warmup_note = "warmup excluded"
            if not run_gpu:
                raise RuntimeError("gpu disabled; pass --gpu to run GPU kernels")

            start = perf_counter()
            gpu_rank, gpu_metrics = run_pagerank_gpu(graph, alpha=alpha, tol=tol, max_iter=max_iter, version=version)
            measured_elapsed = perf_counter() - start
            elapsed = float(gpu_metrics.get("elapsed_seconds", measured_elapsed))
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
                    "warmup_excluded": warmup_gpu,
                    "note": "; ".join(
                        item for item in [warmup_note, str(gpu_metrics.get("fallback_note", ""))] if item
                    ),
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
                    "warmup_excluded": warmup_gpu,
                    "note": f"skipped: {exc}",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU and optional GPU PageRank.")
    parser.add_argument("--config", default="project_spec.yaml")
    parser.add_argument("--graph", default="small")
    parser.add_argument("--edges-path", default=None, help="Optional SNAP-style edge-list path.")
    parser.add_argument("--graph-name", default=None, help="Override graph name in benchmark outputs.")
    parser.add_argument("--gpu", action="store_true", help="Run GPU rows when CUDA is available.")
    parser.add_argument("--versions", default="v1,v2,v3_pull,v3_push")
    parser.add_argument("--sizes", default=None, help="Comma-separated synthetic size names, e.g. small,medium.")
    parser.add_argument("--output", default="artifacts/benchmarks.csv")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--no-scipy-verify", action="store_true")
    parser.add_argument("--no-gpu-warmup", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    alpha, tol, config_max_iter = algorithm_params(config)
    max_iter = int(args.max_iter or config_max_iter)
    versions = _parse_csv_arg(args.versions, ["v1", "v2", "v3_pull", "v3_push"])
    rows: list[dict[str, Any]] = []
    for graph in _load_graphs(args, config):
        rows.extend(
            _benchmark_graph(
                graph,
                alpha=alpha,
                tol=tol,
                max_iter=max_iter,
                versions=versions,
                run_gpu=args.gpu,
                warmup_gpu=not args.no_gpu_warmup,
                verify_scipy=not args.no_scipy_verify,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        "artifacts/benchmark_summary.json",
        {
            "output": str(output_path),
            "rows": rows,
            "cuda_status": cuda_status(),
        },
    )
    if args.gpu:
        write_json(
            "artifacts/gpu_benchmark_summary.json",
            {
                "rows": [row for row in rows if row["backend"] == "gpu"],
                "cuda_status": cuda_status(),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
