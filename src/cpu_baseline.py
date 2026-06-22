from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import algorithm_params, load_config
from src.data_loader import load_edge_list, load_edge_list_with_timings, load_graph_from_config, make_synthetic_graph
from src.metrics import write_json
from src.pagerank_cpu import run_pagerank_cpu


REAL_GRAPH_PATHS = {
    "roadNet-CA": Path("data/graphs/roadNet-CA.tsv"),
    "roadnet-ca": Path("data/graphs/roadNet-CA.tsv"),
    "com-youtube": Path("data/graphs/com-youtube.tsv"),
    "com-Youtube": Path("data/graphs/com-youtube.tsv"),
    "wiki-talk": Path("data/graphs/wiki-talk.tsv"),
    "wiki-Talk": Path("data/graphs/wiki-talk.tsv"),
    "amazon0601": Path("data/graphs/amazon0601.tsv"),
    "soc-livejournal": Path("data/graphs/soc-livejournal.tsv"),
    "soc-LiveJournal1": Path("data/graphs/soc-livejournal.tsv"),
}


def build_graph_with_timings(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    if args.edges:
        return load_edge_list_with_timings(args.edges)
    if args.graph in REAL_GRAPH_PATHS and REAL_GRAPH_PATHS[args.graph].exists():
        return load_edge_list_with_timings(REAL_GRAPH_PATHS[args.graph], remap=False)
    graph_path = Path(args.graph)
    if graph_path.exists():
        return load_edge_list_with_timings(graph_path, remap=False)
    if graph_path.suffix or "/" in args.graph or "\\" in args.graph:
        raise FileNotFoundError(f"Graph path does not exist: {graph_path}")
    if args.graph in {"toy", "chain", "star", "random", "power_law"}:
        return make_synthetic_graph(args.graph), {"load_time_seconds": 0.0, "file_read_time_seconds": 0.0, "csr_build_time_seconds": 0.0}
    return load_graph_from_config(config, graph_size=args.graph), {
        "load_time_seconds": 0.0,
        "file_read_time_seconds": 0.0,
        "csr_build_time_seconds": 0.0,
    }


def build_graph(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    graph, _timings = build_graph_with_timings(args, config)
    return graph


def _print_profile_table(metrics: dict[str, Any]) -> None:
    rows = [
        ("graph loading", metrics.get("load_time_seconds")),
        ("file read", metrics.get("file_read_time_seconds")),
        ("csr build", metrics.get("csr_build_time_seconds")),
        ("dangling mass", metrics.get("dangling_mass_total_seconds")),
        ("spmv", metrics.get("spmv_total_seconds")),
        ("damping", metrics.get("damping_total_seconds")),
        ("normalization", metrics.get("normalization_total_seconds")),
        ("convergence l1", metrics.get("convergence_l1_total_seconds")),
        ("iteration total", metrics.get("total_iteration_time_seconds")),
        ("full convergence", metrics.get("elapsed_seconds")),
    ]
    print("CPU profile timing breakdown")
    print("component          seconds")
    print("----------------  --------")
    for label, value in rows:
        if value is not None:
            print(f"{label:<16}  {float(value):.6f}")
    print(f"SpMV / iteration time: {float(metrics.get('spmv_percent_iteration_time', 0.0)):.2f}%")
    print(f"SpMV / compute time:   {float(metrics.get('spmv_percent_total_compute_time', 0.0)):.2f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CPU PageRank baseline.")
    parser.add_argument("--config", default="project_spec.yaml")
    parser.add_argument("--graph", default="small")
    parser.add_argument("--edges", default=None)
    parser.add_argument("--profile", action="store_true", help="Write detailed CPU timing breakdown evidence.")
    parser.add_argument("--output", default=None, help="Metrics JSON output path.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    alpha, tol, max_iter = algorithm_params(config)
    graph, load_timings = build_graph_with_timings(args, config)
    _, metrics = run_pagerank_cpu(graph, alpha=alpha, tol=tol, max_iter=max_iter)
    metrics["graph_name"] = str(graph["name"])
    metrics["alpha"] = alpha
    metrics["tolerance"] = tol
    metrics["max_iter"] = max_iter
    metrics.update(load_timings)
    metrics["profile"] = bool(args.profile)
    metrics["profile_source"] = "real_snap_graph" if str(graph["name"]) in {"roadNet-CA", "com-youtube", "wiki-talk", "amazon0601", "soc-livejournal"} else "synthetic_or_custom"
    metrics["self_loop_semantics"] = "preserved"
    metrics["duplicate_edge_semantics"] = "preserved"

    output_path = Path(args.output or ("artifacts/profile_summary.json" if args.profile else "artifacts/cpu_baseline_metrics.json"))
    write_json(output_path, metrics)
    write_json("artifacts/cpu_baseline_metrics.json", metrics)
    if "spmv_avg_seconds" in metrics:
        print(f"SpMV avg per-iteration: {metrics['spmv_avg_seconds'] * 1000:.2f} ms")
        print(f"SpMV total (all iters): {metrics['spmv_total_seconds'] * 1000:.2f} ms")
        print(f"Iterations to converge: {metrics['iterations']}")
        print(f"Per-iteration wall time: {metrics['per_iteration_wall_time_seconds'] * 1000:.2f} ms")
    if args.profile:
        _print_profile_table(metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
