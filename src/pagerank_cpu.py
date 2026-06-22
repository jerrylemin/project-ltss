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
) -> tuple[np.ndarray, int, float, bool, dict[str, list[float]]]:
    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), 0, 0.0, True, {}

    rank = np.full(n, 1.0 / n, dtype=np.float64)
    base = (1.0 - alpha) / n
    l1_delta = float("inf")
    converged = False
    timings: dict[str, list[float]] = {
        "dangling_mass_times_seconds": [],
        "spmv_times_seconds": [],
        "damping_times_seconds": [],
        "normalization_times_seconds": [],
        "convergence_l1_times_seconds": [],
        "iteration_wall_times_seconds": [],
    }

    for iteration in range(1, max_iter + 1):
        iteration_start = perf_counter()
        dangling_start = perf_counter()
        dangling_mass = float(rank[out_degree == 0].sum())
        timings["dangling_mass_times_seconds"].append(perf_counter() - dangling_start)
        new_rank = np.empty_like(rank)
        dangling_term = alpha * dangling_mass / n
        spmv_start = perf_counter()
        edge_contrib = rank[indices] / out_degree[indices]
        nonempty = np.diff(indptr) > 0
        new_rank.fill(0.0)
        if edge_contrib.size:
            new_rank[nonempty] = np.add.reduceat(edge_contrib, indptr[:-1][nonempty])
        timings["spmv_times_seconds"].append(perf_counter() - spmv_start)

        damping_start = perf_counter()
        new_rank = base + dangling_term + alpha * new_rank
        timings["damping_times_seconds"].append(perf_counter() - damping_start)

        normalization_start = perf_counter()
        total = float(new_rank.sum())
        if total > 0:
            new_rank /= total
        timings["normalization_times_seconds"].append(perf_counter() - normalization_start)
        convergence_start = perf_counter()
        l1_delta = float(np.abs(new_rank - rank).sum())
        timings["convergence_l1_times_seconds"].append(perf_counter() - convergence_start)
        rank = new_rank
        timings["iteration_wall_times_seconds"].append(perf_counter() - iteration_start)
        if l1_delta < tol:
            converged = True
            break
    return rank, iteration, l1_delta, converged, timings


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
    indptr = np.asarray(graph["indptr"])
    indices = np.asarray(graph["indices"])
    out_degree = np.asarray(graph["out_degree"])

    start = perf_counter()
    timings: dict[str, list[float]] = {}
    if mode == "scipy":
        rank, iterations, l1_delta, converged = _pagerank_scipy_reference(
            indptr, indices, out_degree, alpha, tol, max_iter
        )
    elif mode == "numpy_loop":
        rank, iterations, l1_delta, converged, timings = _pagerank_numpy_loop(
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
    if timings:
        total_iteration_time = float(np.sum(timings["iteration_wall_times_seconds"]))
        total_compute_time = total_iteration_time or elapsed
        for key, values in timings.items():
            metrics[key] = [float(value) for value in values]
            total_key = key.replace("_times_seconds", "_total_seconds")
            avg_key = key.replace("_times_seconds", "_avg_seconds")
            metrics[total_key] = float(np.sum(values))
            metrics[avg_key] = float(np.mean(values))
        metrics["spmv_times_seconds"] = metrics["spmv_times_seconds"]
        metrics["spmv_avg_seconds"] = metrics["spmv_avg_seconds"]
        metrics["spmv_total_seconds"] = metrics["spmv_total_seconds"]
        metrics["per_iteration_wall_time_seconds"] = metrics["iteration_wall_avg_seconds"]
        metrics["total_iteration_time_seconds"] = total_iteration_time
        metrics["total_compute_time_seconds"] = total_compute_time
        metrics["spmv_percent_iteration_time"] = 100.0 * float(metrics["spmv_total_seconds"]) / total_iteration_time if total_iteration_time else 0.0
        metrics["spmv_percent_total_compute_time"] = 100.0 * float(metrics["spmv_total_seconds"]) / total_compute_time if total_compute_time else 0.0

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
