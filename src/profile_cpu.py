from __future__ import annotations

import argparse
import cProfile
import io
import pstats
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import algorithm_params, load_config
from src.data_loader import load_graph_from_config
from src.metrics import write_json
from src.pagerank_cpu import run_pagerank_cpu


def _top_functions(stats: pstats.Stats, limit: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (filename, line, func), stat in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:limit]:
        ccalls, ncalls, total_time, cumulative_time, _callers = stat
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{func}",
                "primitive_calls": int(ccalls),
                "total_calls": int(ncalls),
                "total_time": float(total_time),
                "cumulative_time": float(cumulative_time),
            }
        )
    return rows


def _write_bottleneck_doc(summary: dict[str, object]) -> None:
    Path("docs").mkdir(exist_ok=True)
    lines = [
        "# Bottleneck Decision",
        "",
        f"Last updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The dominant bottleneck is the PageRank iteration SpMV step: each iteration scans the incoming CSR structure and gathers `rank[src] / out_degree[src]` for every edge.",
        "",
        "This makes the project a good fit for partial GPU acceleration because the graph traversal and vector update are repeated many times while configuration, loading, validation, and reporting remain on CPU.",
        "",
        "## Profile Summary",
        "",
        f"- Graph: {summary.get('graph_name')}",
        f"- Nodes: {summary.get('num_nodes')}",
        f"- Edges: {summary.get('num_edges')}",
        f"- Elapsed seconds: {summary.get('elapsed_seconds')}",
        f"- Iterations: {summary.get('iterations')}",
        "",
        "## Top Functions by Cumulative Time",
        "",
    ]
    for item in summary.get("top_functions", []):
        lines.append(
            f"- `{item['function']}`: cumtime={item['cumulative_time']:.6f}s, calls={item['total_calls']}"
        )
    Path("docs/bottleneck_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile CPU PageRank baseline.")
    parser.add_argument("--config", default="project_spec.yaml")
    parser.add_argument("--graph", default="small")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    alpha, tol, max_iter = algorithm_params(config)
    graph = load_graph_from_config(config, graph_size=args.graph)

    profiler = cProfile.Profile()
    profiler.enable()
    _rank, metrics = run_pagerank_cpu(
        graph, alpha=alpha, tol=tol, max_iter=max_iter, verify_scipy=False
    )
    profiler.disable()

    Path("artifacts").mkdir(exist_ok=True)
    profiler.dump_stats("artifacts/cpu_profile.prof")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(10)

    summary = {
        **metrics,
        "graph_name": graph["name"],
        "profile_path": "artifacts/cpu_profile.prof",
        "top_functions": _top_functions(stats, 10),
    }
    write_json("artifacts/profile_summary.json", summary)
    _write_bottleneck_doc(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
