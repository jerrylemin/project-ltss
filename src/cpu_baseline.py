from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import algorithm_params, load_config
from src.data_loader import load_edge_list, load_graph_from_config, make_synthetic_graph
from src.metrics import write_json
from src.pagerank_cpu import run_pagerank_cpu


def build_graph(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.edges:
        return load_edge_list(args.edges)
    if args.graph in {"toy", "chain", "star", "random", "power_law"}:
        return make_synthetic_graph(args.graph)
    return load_graph_from_config(config, graph_size=args.graph)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CPU PageRank baseline.")
    parser.add_argument("--config", default="project_spec.yaml")
    parser.add_argument("--graph", default="small")
    parser.add_argument("--edges", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    alpha, tol, max_iter = algorithm_params(config)
    graph = build_graph(args, config)
    _, metrics = run_pagerank_cpu(graph, alpha=alpha, tol=tol, max_iter=max_iter)
    metrics["graph_name"] = str(graph["name"])
    metrics["alpha"] = alpha
    metrics["tolerance"] = tol
    metrics["max_iter"] = max_iter

    output_path = Path("artifacts/cpu_baseline_metrics.json")
    write_json(output_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
