from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import algorithm_params, load_config
from src.data_loader import load_edge_list, load_graph_from_config
from src.gpu.cuda_utils import cuda_available, cuda_status
from src.gpu.pagerank_gpu import run_pagerank_gpu
from src.metrics import relative_l1_error, write_json
from src.pagerank_cpu import run_pagerank_cpu


BENCHMARK_GRAPHS = [
    ("roadNet-CA", "data/graphs/roadNet-CA.tsv"),
    ("com-youtube", "data/graphs/com-youtube.tsv"),
    ("wiki-talk", "data/graphs/wiki-talk.tsv"),
    ("amazon0601", "data/graphs/amazon0601.tsv"),
    ("soc-livejournal", "data/graphs/soc-livejournal.tsv"),
]

VERSIONS = ["cpu_numpy", "gpu_v1", "gpu_v2", "gpu_v3_pull", "gpu_v3_push"]

GPU_VERSION_MAP = {
    "v1": "v1",
    "v2": "v2",
    "v3_pull": "v3_pull",
    "v3_push": "v3_push",
    "gpu_v1": "v1",
    "gpu_v2": "v2",
    "gpu_v3_pull": "v3_pull",
    "gpu_v3_push": "v3_push",
}

VERSION_ALIASES = {
    "v1": "gpu_v1",
    "v2": "gpu_v2",
    "v3_pull": "gpu_v3_pull",
    "v3_push": "gpu_v3_push",
    "numpy_loop": "cpu_numpy",
}

COLUMNS = [
    "graph_name",
    "version",
    "n_nodes",
    "n_edges",
    "convergence_time_s",
    "iterations",
    "iter_per_sec",
    "cpu_spmv_total_s",
    "version_time_over_cpu",
    "speedup_vs_cpu",
    "relative_l1_vs_scipy",
    "spearman_vs_scipy",
    "target_under_5s",
    "cuda_available",
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


def _normalize_versions(versions: list[str]) -> list[str]:
    return [VERSION_ALIASES.get(version, version) for version in versions]


def _spearman_top_k(actual: np.ndarray, expected: np.ndarray, k: int = 1000) -> float | None:
    if actual.size == 0 or expected.size == 0:
        return None
    top_k = min(k, expected.size)
    top_idx = np.argsort(expected)[-top_k:]
    actual_values = actual[top_idx]
    expected_values = expected[top_idx]
    if np.allclose(actual_values, expected_values):
        return 1.0
    try:
        from scipy.stats import spearmanr

        correlation = spearmanr(expected_values, actual_values).correlation
        return None if np.isnan(correlation) else float(correlation)
    except Exception:
        return None


def _load_benchmark_graphs(args: argparse.Namespace, config: dict[str, Any]) -> list[tuple[str, str | None, dict[str, Any] | None]]:
    if args.edges_path:
        graph = load_edge_list(args.edges_path, remap=False)
        graph_name = args.graph_name or str(graph["name"])
        return [(graph_name, args.edges_path, {**graph, "name": graph_name})]

    if args.synthetic:
        graph_sizes = _parse_csv_arg(args.sizes, [args.graph])
        graphs = []
        for size in graph_sizes:
            graph = load_graph_from_config(config, graph_size=size)
            graphs.append((str(graph["name"]), None, graph))
        return graphs

    graphs: list[tuple[str, str | None, dict[str, Any] | None]] = []
    for graph_name, path in BENCHMARK_GRAPHS:
        graph_path = Path(path)
        if graph_path.exists():
            graph = load_edge_list(graph_path, remap=False)
            graphs.append((graph_name, path, {**graph, "name": graph_name}))
        else:
            graphs.append((graph_name, path, None))
    return graphs


def run_pagerank(
    graph: dict[str, Any],
    version: str,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha: float = 0.85,
) -> tuple[np.ndarray, dict[str, Any]]:
    if version == "cpu_numpy":
        rank, metrics = run_pagerank_cpu(
            graph,
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            mode="numpy_loop",
            verify_scipy=False,
        )
        return rank, {**metrics, "mode": "cpu_numpy"}
    if version in GPU_VERSION_MAP:
        return run_pagerank_gpu(graph, alpha=alpha, tol=tol, max_iter=max_iter, version=GPU_VERSION_MAP[version])
    raise ValueError(f"Unknown benchmark version: {version}")


def _benchmark_graph(
    graph_name: str,
    path: str | None,
    graph: dict[str, Any] | None,
    *,
    alpha: float,
    tol: float,
    max_iter: int,
    versions: list[str],
) -> list[dict[str, Any]]:
    if graph is None:
        return [
            {
                "graph_name": graph_name,
                "version": version,
                "n_nodes": None,
                "n_edges": None,
                "convergence_time_s": None,
                "iterations": None,
                "iter_per_sec": None,
                "cpu_spmv_total_s": None,
                "version_time_over_cpu": None,
                "speedup_vs_cpu": None,
                "relative_l1_vs_scipy": None,
                "spearman_vs_scipy": None,
                "target_under_5s": None,
                "cuda_available": cuda_available(),
                "note": f"missing graph file: {path}; run python scripts/download_graphs.py",
            }
            for version in versions
        ]

    scipy_rank, scipy_metrics = run_pagerank_cpu(
        graph,
        alpha=alpha,
        tol=tol,
        max_iter=max_iter,
        mode="scipy",
        verify_scipy=False,
    )

    rows: list[dict[str, Any]] = []
    cpu_elapsed: float | None = None
    cpu_spmv_total: float | None = None
    for version in versions:
        try:
            if version.startswith("gpu_") and not cuda_available():
                raise RuntimeError("skipped: no cuda")
            start = perf_counter()
            rank, metrics = run_pagerank(graph, version, tol=tol, max_iter=max_iter, alpha=alpha)
            measured_elapsed = perf_counter() - start
            elapsed = float(metrics.get("elapsed_seconds", measured_elapsed))
            if version == "cpu_numpy":
                cpu_elapsed = elapsed
                cpu_spmv_total = metrics.get("spmv_total_seconds")
            rows.append(
                {
                    "graph_name": graph_name,
                    "version": version,
                    "n_nodes": int(graph["num_nodes"]),
                    "n_edges": int(graph["num_edges"]),
                    "convergence_time_s": elapsed,
                    "iterations": int(metrics.get("iterations", 0)),
                    "iter_per_sec": _iterations_per_second(metrics.get("iterations"), elapsed),
                    "cpu_spmv_total_s": float(metrics["spmv_total_seconds"]) if "spmv_total_seconds" in metrics else None,
                    "version_time_over_cpu": (elapsed / cpu_elapsed) if cpu_elapsed else None,
                    "speedup_vs_cpu": (cpu_elapsed / elapsed) if cpu_elapsed and elapsed else None,
                    "relative_l1_vs_scipy": relative_l1_error(rank, scipy_rank),
                    "spearman_vs_scipy": _spearman_top_k(rank, scipy_rank),
                    "target_under_5s": graph_name == "com-youtube" and elapsed <= 5.0,
                    "cuda_available": cuda_available(),
                    "note": str(metrics.get("fallback_note", "")),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "graph_name": graph_name,
                    "version": version,
                    "n_nodes": int(graph["num_nodes"]),
                    "n_edges": int(graph["num_edges"]),
                    "convergence_time_s": None,
                    "iterations": None,
                    "iter_per_sec": None,
                    "cpu_spmv_total_s": None,
                    "version_time_over_cpu": None,
                    "speedup_vs_cpu": None,
                    "relative_l1_vs_scipy": None,
                    "spearman_vs_scipy": None,
                    "target_under_5s": None,
                    "cuda_available": cuda_available(),
                    "note": str(exc),
                }
            )

    for row in rows:
        if row["version"] != "cpu_numpy" and row["convergence_time_s"] is not None and cpu_elapsed:
            row["version_time_over_cpu"] = float(row["convergence_time_s"]) / cpu_elapsed
            row["speedup_vs_cpu"] = cpu_elapsed / float(row["convergence_time_s"])
        if row["version"] != "cpu_numpy" and cpu_spmv_total:
            row["cpu_spmv_total_s"] = cpu_spmv_total
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["graph_name", "version", "n_nodes", "n_edges", "convergence_time_s", "iterations", "iter_per_sec", "speedup_vs_cpu", "spearman_vs_scipy", "note"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            value = row.get(header)
            text = "" if value is None else str(value)
            widths[header] = min(max(widths[header], len(text)), 32)

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = "" if value is None else str(value)
            if len(text) > widths[header]:
                text = text[: widths[header] - 1] + "."
            cells.append(text.ljust(widths[header]))
        print(" | ".join(cells))


def _print_com_youtube_target(rows: list[dict[str, Any]]) -> None:
    gpu_rows = [
        row
        for row in rows
        if row["graph_name"] == "com-youtube" and str(row["version"]).startswith("gpu_") and row["convergence_time_s"] is not None
    ]
    if not gpu_rows:
        print("com-youtube target: NOT MEASURED -- no successful GPU row.")
        return

    best = min(gpu_rows, key=lambda row: float(row["convergence_time_s"]))
    elapsed = float(best["convergence_time_s"])
    if elapsed <= 5.0:
        print(f"TARGET MET: com-youtube best={best['version']} time={elapsed:.6f}s")
    else:
        print(f"TARGET NOT MET - gap: {elapsed - 5.0:.6f}s over budget")


def _print_speedup_ratios(rows: list[dict[str, Any]]) -> None:
    graph_names = list(dict.fromkeys(str(row["graph_name"]) for row in rows))
    for graph_name in graph_names:
        graph_rows = [row for row in rows if row["graph_name"] == graph_name]
        by_version = {str(row["version"]): row for row in graph_rows}
        v1_time = by_version.get("gpu_v1", {}).get("convergence_time_s")
        v2_time = by_version.get("gpu_v2", {}).get("convergence_time_s")
        pull_time = by_version.get("gpu_v3_pull", {}).get("convergence_time_s")
        push_time = by_version.get("gpu_v3_push", {}).get("convergence_time_s")
        v3_times = [float(value) for value in (pull_time, push_time) if value]
        cpu_spmv_time = by_version.get("cpu_numpy", {}).get("cpu_spmv_total_s")

        print(f"{graph_name} speedup ratios:")
        if v1_time and v2_time:
            print(f"V2 vs V1 speedup: {float(v1_time) / float(v2_time):.1f}x")
        if v2_time and v3_times:
            print(f"V3 vs V2 speedup: {float(v2_time) / min(v3_times):.1f}x")
        if cpu_spmv_time and v3_times:
            print(f"Best GPU vs CPU SpMV speedup: {float(cpu_spmv_time) / min(v3_times):.1f}x")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark CPU and GPU PageRank on the required SNAP graphs.")
    parser.add_argument("--config", default="project_spec.yaml")
    parser.add_argument("--graph", default="small", help="Synthetic graph size when --synthetic is used.")
    parser.add_argument("--sizes", default=None, help="Comma-separated synthetic sizes when --synthetic is used.")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic config graphs instead of the five SNAP graphs.")
    parser.add_argument("--edges-path", default=None, help="Optional single SNAP-style edge-list path.")
    parser.add_argument("--graph-name", default=None, help="Override graph name for --edges-path.")
    parser.add_argument("--versions", default=",".join(VERSIONS), help="Comma-separated versions to run.")
    parser.add_argument("--output", default="artifacts/benchmark_results.csv")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="Compatibility flag; GPU versions are controlled by --versions.")
    parser.add_argument("--no-scipy-verify", action="store_true", help="Compatibility flag; full benchmark keeps SciPy reference enabled.")
    parser.add_argument("--no-gpu-warmup", action="store_true", help="Compatibility flag retained for older scripts.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    alpha, config_tol, config_max_iter = algorithm_params(config)
    tol = min(float(config_tol), 1e-6)
    max_iter = int(args.max_iter or config_max_iter)
    versions = _normalize_versions(_parse_csv_arg(args.versions, VERSIONS))

    rows: list[dict[str, Any]] = []
    for graph_name, path, graph in _load_benchmark_graphs(args, config):
        rows.extend(
            _benchmark_graph(
                graph_name,
                path,
                graph,
                alpha=alpha,
                tol=tol,
                max_iter=max_iter,
                versions=versions,
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
            "benchmark_graphs": BENCHMARK_GRAPHS,
            "versions": versions,
            "tolerance": tol,
            "max_iter": max_iter,
        },
    )

    _print_table(rows)
    _print_speedup_ratios(rows)
    _print_com_youtube_target(rows)
    print(f"Saved benchmark table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
