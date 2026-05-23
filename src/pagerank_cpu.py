from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


def _pagerank_numpy_loop(
    indptr: np.ndarray,
    indices: np.ndarray,
    out_degree: np.ndarray,
    alpha: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float, bool]:
    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), 0, 0.0, True

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    base = (1.0 - alpha) / n
    l1_delta = float("inf")
    converged = False

    for iteration in range(1, max_iter + 1):
        dangling_mass = float(rank[out_degree == 0].sum())
        new_rank = np.empty_like(rank)
        dangling_term = alpha * dangling_mass / n
        for node in range(n):
            start = indptr[node]
            end = indptr[node + 1]
            incoming = indices[start:end]
            if incoming.size:
                contribution = np.sum(rank[incoming] / out_degree[incoming])
            else:
                contribution = 0.0
            new_rank[node] = base + dangling_term + alpha * contribution

        total = float(new_rank.sum())
        if total > 0:
            new_rank /= total
        l1_delta = float(np.abs(new_rank - rank).sum())
        rank = new_rank
        if l1_delta < tol:
            converged = True
            break
    return rank, iteration, l1_delta, converged


def _pagerank_scipy_reference(
    indptr: np.ndarray,
    indices: np.ndarray,
    out_degree: np.ndarray,
    alpha: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float, bool]:
    from scipy.sparse import csr_matrix

    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), 0, 0.0, True

    row_counts = np.diff(indptr)
    rows = np.repeat(np.arange(n, dtype=np.int64), row_counts)
    safe_out = np.maximum(out_degree[indices], 1)
    data = 1.0 / safe_out.astype(np.float64)
    matrix = csr_matrix((data, indices, indptr), shape=(n, n))

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    base = (1.0 - alpha) / n
    l1_delta = float("inf")
    converged = False
    del rows

    for iteration in range(1, max_iter + 1):
        dangling_mass = float(rank[out_degree == 0].sum())
        new_rank = base + alpha * (matrix @ rank + dangling_mass / n)
        total = float(new_rank.sum())
        if total > 0:
            new_rank /= total
        l1_delta = float(np.abs(new_rank - rank).sum())
        rank = np.asarray(new_rank, dtype=np.float64)
        if l1_delta < tol:
            converged = True
            break
    return rank, iteration, l1_delta, converged


def run_pagerank_cpu(
    graph: dict[str, Any],
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 200,
    mode: str = "numpy_loop",
    verify_scipy: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    indptr = np.asarray(graph["indptr"], dtype=np.int64)
    indices = np.asarray(graph["indices"], dtype=np.int64)
    out_degree = np.asarray(graph["out_degree"], dtype=np.int64)

    start = perf_counter()
    if mode == "scipy":
        rank, iterations, l1_delta, converged = _pagerank_scipy_reference(
            indptr, indices, out_degree, alpha, tol, max_iter
        )
    elif mode == "numpy_loop":
        rank, iterations, l1_delta, converged = _pagerank_numpy_loop(
            indptr, indices, out_degree, alpha, tol, max_iter
        )
    else:
        raise ValueError(f"Unknown CPU PageRank mode: {mode}")
    elapsed = perf_counter() - start

    metrics: dict[str, Any] = {
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "iterations": int(iterations),
        "elapsed_seconds": float(elapsed),
        "l1_delta": float(l1_delta),
        "rank_sum": float(rank.sum()) if rank.size else 0.0,
        "converged": bool(converged),
        "mode": mode,
    }

    if verify_scipy and mode != "scipy":
        try:
            ref_rank, *_ = _pagerank_scipy_reference(indptr, indices, out_degree, alpha, tol, max_iter)
            l1_error = float(np.abs(rank - ref_rank).sum())
            denominator = float(np.abs(ref_rank).sum()) or 1.0
            metrics["l1_error_vs_scipy"] = l1_error
            metrics["relative_error_vs_scipy"] = l1_error / denominator
        except Exception as exc:  # pragma: no cover - depends on optional SciPy install
            metrics["scipy_reference_error"] = str(exc)

    return rank, metrics
