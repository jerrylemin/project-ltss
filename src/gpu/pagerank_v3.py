from __future__ import annotations

from time import perf_counter

import numpy as np

from src.gpu.cuda_utils import require_cuda


def cuda_available() -> bool:
    from src.gpu.cuda_utils import cuda_available as _cuda_available

    return _cuda_available()


def _edge_arrays_from_incoming(graph: dict) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.asarray(graph["indptr"], dtype=np.int64)
    indices = np.asarray(graph["indices"], dtype=np.int64)
    dst = np.repeat(np.arange(int(graph["num_nodes"]), dtype=np.int64), np.diff(indptr))
    src = indices.copy()
    return src, dst


def run(graph: dict, alpha: float, tol: float, max_iter: int, mode: str = "pull_gather") -> tuple[np.ndarray, dict]:
    require_cuda()
    if mode not in {"pull_gather", "push_scatter"}:
        raise ValueError(f"Unknown v3 mode: {mode}")
    if mode == "pull_gather":
        from src.gpu.pagerank_v2 import run as pull_run

        rank, metrics = pull_run(graph, alpha, tol, max_iter)
        metrics["mode"] = "gpu_v3_pull_gather"
        metrics["fallback_note"] = "pull_gather reuses the fused v2 kernel for stable correctness."
        return rank, metrics

    from numba import cuda

    @cuda.jit
    def push_scatter_kernel(src_nodes, dst_nodes, out_degree, old_rank, accum):
        """One CUDA thread owns one edge and scatters contribution with atomic add to destination rank."""
        edge = cuda.grid(1)
        if edge < src_nodes.size:
            src = src_nodes[edge]
            degree = out_degree[src]
            if degree > 0:
                cuda.atomic.add(accum, dst_nodes[edge], old_rank[src] / degree)

    @cuda.jit
    def damping_kernel(accum, out_degree, old_rank, new_rank, alpha_value, dangling_mass, l1_delta):
        """One CUDA thread completes damping per node; L1 uses atomic float reduction as Numba allows."""
        node = cuda.grid(1)
        n = old_rank.size
        if node < n:
            value = (1.0 - alpha_value) / n + alpha_value * (accum[node] + dangling_mass / n)
            new_rank[node] = value
            cuda.atomic.add(l1_delta, 0, abs(value - old_rank[node]))

    out_degree = np.asarray(graph["out_degree"], dtype=np.int64)
    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), {"iterations": 0, "elapsed_seconds": 0.0, "converged": True}
    src, dst = _edge_arrays_from_incoming(graph)

    start = perf_counter()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    d_src = cuda.to_device(src)
    d_dst = cuda.to_device(dst)
    d_out_degree = cuda.to_device(out_degree)
    d_rank = cuda.to_device(rank)
    d_new_rank = cuda.device_array_like(d_rank)
    threads = 128
    edge_blocks = (len(src) + threads - 1) // threads
    node_blocks = (n + threads - 1) // threads
    converged = False
    l1_delta = float("inf")

    for iteration in range(1, max_iter + 1):
        dangling_mass = float(rank[out_degree == 0].sum())
        d_accum = cuda.to_device(np.zeros(n, dtype=np.float64))
        d_delta = cuda.to_device(np.array([0.0], dtype=np.float64))
        push_scatter_kernel[edge_blocks, threads](d_src, d_dst, d_out_degree, d_rank, d_accum)
        damping_kernel[node_blocks, threads](d_accum, d_out_degree, d_rank, d_new_rank, alpha, dangling_mass, d_delta)
        cuda.synchronize()
        rank = d_new_rank.copy_to_host()
        total = float(rank.sum())
        if total > 0:
            rank /= total
        l1_delta = float(d_delta.copy_to_host()[0])
        d_rank = cuda.to_device(rank)
        if l1_delta < tol:
            converged = True
            break

    elapsed = perf_counter() - start
    return rank, {
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "iterations": int(iteration),
        "elapsed_seconds": float(elapsed),
        "l1_delta": float(l1_delta),
        "rank_sum": float(rank.sum()),
        "converged": bool(converged),
        "mode": "gpu_v3_push_scatter",
        "fallback_note": "If atomic float support is unavailable on the device, caller should skip this mode.",
    }


def benchmark_modes(graph: dict, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 200) -> list[dict]:
    rows = []
    for mode in ("pull_gather", "push_scatter"):
        try:
            _rank, metrics = run(graph, alpha, tol, max_iter, mode=mode)
            rows.append({"mode": mode, **metrics})
        except RuntimeError as exc:
            rows.append({"mode": mode, "elapsed_seconds": None, "note": f"skipped: {exc}"})
    return rows
