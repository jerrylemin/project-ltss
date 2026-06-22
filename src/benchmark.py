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
from src.data_loader import load_edge_list, load_edge_list_with_timings, load_graph_from_config
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
    "cpu": "cpu_numpy",
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
    "repeat_count",
    "warmup_count",
    "load_time_s",
    "preprocess_time_s",
    "h2d_time_s",
    "kernel_time_s",
    "d2h_time_s",
    "total_time_s",
    "convergence_time_mean_s",
    "convergence_time_median_s",
    "convergence_time_min_s",
    "convergence_time_max_s",
    "convergence_time_std_s",
    "total_time_mean_s",
    "total_time_median_s",
    "total_time_min_s",
    "total_time_max_s",
    "total_time_std_s",
    "h2d_time_median_s",
    "kernel_time_median_s",
    "d2h_time_median_s",
    "cpu_spmv_median_s",
    "speedup_vs_cpu_spmv",
    "speedup_vs_cpu_full",
    "include_transfer_timing",
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


def _stat(values: list[float], name: str) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if name == "mean":
        return float(np.mean(arr))
    if name == "median":
        return float(np.median(arr))
    if name == "min":
        return float(np.min(arr))
    if name == "max":
        return float(np.max(arr))
    if name == "std":
        return float(np.std(arr))
    raise ValueError(f"Unknown statistic: {name}")


def _graph_filter_names(value: str) -> set[str] | None:
    if value.lower() == "all":
        return None
    aliases = {
        "com-youtube": "com-youtube",
        "youtube": "com-youtube",
        "roadnet-ca": "roadNet-CA",
        "roadnet": "roadNet-CA",
        "wiki-talk": "wiki-talk",
        "wikitalk": "wiki-talk",
        "amazon0601": "amazon0601",
        "soc-livejournal": "soc-livejournal",
        "soc-livejournal1": "soc-livejournal",
        "livejournal": "soc-livejournal",
    }
    selected = set()
    for raw_name in _parse_csv_arg(value, []):
        key = raw_name.lower()
        selected.add(aliases.get(key, raw_name))
    return selected


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
        graph, load_timings = load_edge_list_with_timings(args.edges_path, remap=False)
        graph_name = args.graph_name or str(graph["name"])
        return [(graph_name, args.edges_path, {**graph, "name": graph_name, "_load_timings": load_timings})]

    if args.synthetic:
        graph_sizes = _parse_csv_arg(args.sizes, [args.graph])
        graphs = []
        for size in graph_sizes:
            graph = load_graph_from_config(config, graph_size=size)
            graphs.append((str(graph["name"]), None, {**graph, "_load_timings": {}}))
        return graphs

    selected = _graph_filter_names(args.graphs)
    graphs: list[tuple[str, str | None, dict[str, Any] | None]] = []
    for graph_name, path in BENCHMARK_GRAPHS:
        if selected is not None and graph_name not in selected:
            continue
        graph_path = Path(path)
        if graph_path.exists():
            graph, load_timings = load_edge_list_with_timings(graph_path, remap=False)
            graphs.append((graph_name, path, {**graph, "name": graph_name, "_load_timings": load_timings}))
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
    repeat: int,
    warmup: int,
    include_transfer_timing: bool,
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
                "repeat_count": 0,
                "warmup_count": warmup,
                "include_transfer_timing": include_transfer_timing,
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
    load_timings = graph.get("_load_timings", {}) if graph else {}
    for version in versions:
        try:
            if version.startswith("gpu_") and not cuda_available():
                raise RuntimeError("skipped: no cuda")
            if version.startswith("gpu_"):
                for _ in range(max(0, warmup)):
                    run_pagerank(graph, version, tol=tol, max_iter=max_iter, alpha=alpha)
            convergence_times: list[float] = []
            total_times: list[float] = []
            h2d_times: list[float] = []
            kernel_times: list[float] = []
            d2h_times: list[float] = []
            cpu_spmv_times: list[float] = []
            rank = None
            metrics: dict[str, Any] = {}
            for _ in range(max(1, repeat)):
                start = perf_counter()
                rank, metrics = run_pagerank(graph, version, tol=tol, max_iter=max_iter, alpha=alpha)
                measured_elapsed = perf_counter() - start
                elapsed = float(metrics.get("elapsed_seconds", measured_elapsed))
                convergence_times.append(float(metrics.get("convergence_time_seconds", elapsed)))
                total_times.append(float(metrics.get("total_time_seconds", elapsed)))
                h2d_times.append(float(metrics.get("h2d_time_seconds", 0.0)))
                kernel_times.append(float(metrics.get("kernel_time_seconds", metrics.get("total_iteration_time_seconds", elapsed))))
                d2h_times.append(float(metrics.get("d2h_time_seconds", 0.0)))
                if "spmv_total_seconds" in metrics:
                    cpu_spmv_times.append(float(metrics["spmv_total_seconds"]))
            assert rank is not None
            convergence_median = _stat(convergence_times, "median")
            total_median = _stat(total_times, "median")
            cpu_spmv_median = _stat(cpu_spmv_times, "median")
            rows.append(
                {
                    "graph_name": graph_name,
                    "version": version,
                    "n_nodes": int(graph["num_nodes"]),
                    "n_edges": int(graph["num_edges"]),
                    "convergence_time_s": convergence_median,
                    "iterations": int(metrics.get("iterations", 0)),
                    "iter_per_sec": _iterations_per_second(metrics.get("iterations"), convergence_median),
                    "cpu_spmv_total_s": cpu_spmv_median,
                    "version_time_over_cpu": None,
                    "speedup_vs_cpu": None,
                    "relative_l1_vs_scipy": relative_l1_error(rank, scipy_rank),
                    "spearman_vs_scipy": _spearman_top_k(rank, scipy_rank),
                    "target_under_5s": graph_name == "com-youtube" and bool(convergence_median is not None and convergence_median <= 5.0),
                    "cuda_available": cuda_available(),
                    "note": str(metrics.get("fallback_note", "")),
                    "repeat_count": int(max(1, repeat)),
                    "warmup_count": int(warmup if version.startswith("gpu_") else 0),
                    "load_time_s": load_timings.get("load_time_seconds"),
                    "preprocess_time_s": load_timings.get("csr_build_time_seconds"),
                    "h2d_time_s": _stat(h2d_times, "median"),
                    "kernel_time_s": _stat(kernel_times, "median"),
                    "d2h_time_s": _stat(d2h_times, "median"),
                    "total_time_s": total_median,
                    "convergence_time_mean_s": _stat(convergence_times, "mean"),
                    "convergence_time_median_s": convergence_median,
                    "convergence_time_min_s": _stat(convergence_times, "min"),
                    "convergence_time_max_s": _stat(convergence_times, "max"),
                    "convergence_time_std_s": _stat(convergence_times, "std"),
                    "total_time_mean_s": _stat(total_times, "mean"),
                    "total_time_median_s": total_median,
                    "total_time_min_s": _stat(total_times, "min"),
                    "total_time_max_s": _stat(total_times, "max"),
                    "total_time_std_s": _stat(total_times, "std"),
                    "h2d_time_median_s": _stat(h2d_times, "median"),
                    "kernel_time_median_s": _stat(kernel_times, "median"),
                    "d2h_time_median_s": _stat(d2h_times, "median"),
                    "cpu_spmv_median_s": cpu_spmv_median,
                    "speedup_vs_cpu_spmv": None,
                    "speedup_vs_cpu_full": None,
                    "include_transfer_timing": include_transfer_timing,
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
                    "repeat_count": 0,
                    "warmup_count": warmup,
                    "include_transfer_timing": include_transfer_timing,
                }
            )

    cpu_rows = [row for row in rows if row["version"] == "cpu_numpy" and row.get("convergence_time_s")]
    cpu_full = float(cpu_rows[0]["convergence_time_s"]) if cpu_rows else None
    cpu_spmv = float(cpu_rows[0]["cpu_spmv_median_s"]) if cpu_rows and cpu_rows[0].get("cpu_spmv_median_s") else None
    for row in rows:
        if row["version"] != "cpu_numpy" and row.get("convergence_time_s") is not None:
            elapsed = float(row["convergence_time_s"])
            if cpu_full:
                row["version_time_over_cpu"] = elapsed / cpu_full
                row["speedup_vs_cpu"] = cpu_full / elapsed
                row["speedup_vs_cpu_full"] = cpu_full / elapsed
            if cpu_spmv:
                row["cpu_spmv_total_s"] = cpu_spmv
                row["cpu_spmv_median_s"] = cpu_spmv
                row["speedup_vs_cpu_spmv"] = cpu_spmv / elapsed
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
    parser.add_argument("--graphs", default="all", help="'all' or comma-separated graph names from the five SNAP targets.")
    parser.add_argument("--output", default="artifacts/benchmark_results.csv")
    parser.add_argument("--repeat", type=int, default=5, help="Timed repeats per graph/version.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed CUDA warmup runs per GPU graph/version.")
    parser.add_argument("--no-write-default-artifacts", action="store_true", help="Only write the requested CSV output, not benchmark_summary.json.")
    parser.add_argument("--include-transfer-timing", action="store_true", help="Include H2D/kernel/D2H timing columns in the CSV.")
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
                repeat=max(1, args.repeat),
                warmup=max(0, args.warmup),
                include_transfer_timing=bool(args.include_transfer_timing),
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if not args.no_write_default_artifacts:
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
                "repeat": max(1, args.repeat),
                "warmup": max(0, args.warmup),
                "self_loop_semantics": "preserved",
                "duplicate_edge_semantics": "preserved",
            },
        )

    _print_table(rows)
    _print_speedup_ratios(rows)
    _print_com_youtube_target(rows)
    print(f"Saved benchmark table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
