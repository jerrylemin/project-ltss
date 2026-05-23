from __future__ import annotations

from typing import Any

import numpy as np


def run_pagerank_gpu(
    graph: dict,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
    version: str = "v1",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (rank, metrics) with the same shape as CPU output."""
    if version == "v1":
        from src.gpu.pagerank_v1 import run

        return run(graph, alpha, tol, max_iter)
    if version == "v2":
        from src.gpu.pagerank_v2 import run

        return run(graph, alpha, tol, max_iter)
    if version == "v3_pull":
        from src.gpu.pagerank_v3 import run

        return run(graph, alpha, tol, max_iter, mode="pull_gather")
    if version == "v3_push":
        from src.gpu.pagerank_v3 import run

        return run(graph, alpha, tol, max_iter, mode="push_scatter")
    raise ValueError(f"Unknown GPU PageRank version: {version}")
