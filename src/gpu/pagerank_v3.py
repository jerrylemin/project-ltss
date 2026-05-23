from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from src.gpu.cuda_utils import require_cuda

_PULL_KERNELS: tuple[Callable, Callable, Callable, Callable] | None = None


def cuda_available() -> bool:
    from src.gpu.cuda_utils import cuda_available as _cuda_available

    return _cuda_available()


def _edge_arrays_from_incoming(graph: dict) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.asarray(graph["indptr"], dtype=np.int64)
    indices = np.asarray(graph["indices"], dtype=np.int64)
    dst = np.repeat(np.arange(int(graph["num_nodes"]), dtype=np.int64), np.diff(indptr))
    src = indices.copy()
    return src, dst


def _pull_kernels() -> tuple[Callable, Callable, Callable, Callable]:
    global _PULL_KERNELS
    if _PULL_KERNELS is not None:
        return _PULL_KERNELS

    from numba import cuda

    @cuda.jit
    def reset_stats_kernel(stats):
        """Clear dangling mass, L1 delta, and rank sum accumulators before one iteration."""
        if cuda.grid(1) == 0:
            stats[0] = 0.0
            stats[1] = 0.0
            stats[2] = 0.0

    @cuda.jit
    def dangling_mass_kernel(out_degree, rank, stats):
        """One thread checks one node and atomically accumulates rank mass for dangling nodes."""
        node = cuda.grid(1)
        if node < out_degree.size and out_degree[node] == 0:
            cuda.atomic.add(stats, 0, rank[node])

    @cuda.jit
    def pull_update_kernel(indptr, indices, out_degree, old_rank, new_rank, stats, alpha_value):
        """One thread gathers incoming CSR edges, applies damping, and reduces L1/rank sum on device."""
        node = cuda.grid(1)
        n = old_rank.size
        if node < n:
            total = 0.0
            for offset in range(indptr[node], indptr[node + 1]):
                src = indices[offset]
                degree = out_degree[src]
                if degree > 0:
                    total += old_rank[src] / degree
            value = (1.0 - alpha_value) / n + alpha_value * (total + stats[0] / n)
            new_rank[node] = value
            cuda.atomic.add(stats, 1, abs(value - old_rank[node]))
            cuda.atomic.add(stats, 2, value)

    @cuda.jit
    def normalize_kernel(rank, rank_sum):
        """One thread normalizes one rank entry after device-side rank sum reduction."""
        node = cuda.grid(1)
        if node < rank.size and rank_sum > 0.0:
            rank[node] = rank[node] / rank_sum

    _PULL_KERNELS = (reset_stats_kernel, dangling_mass_kernel, pull_update_kernel, normalize_kernel)
    return _PULL_KERNELS


def _run_pull_gather(graph: dict, alpha: float, tol: float, max_iter: int) -> tuple[np.ndarray, dict]:
    require_cuda()
    from numba import cuda

    reset_stats_kernel, dangling_mass_kernel, pull_update_kernel, normalize_kernel = _pull_kernels()

    indptr = np.asarray(graph["indptr"], dtype=np.int64)
    indices = np.asarray(graph["indices"], dtype=np.int64)
    out_degree = np.asarray(graph["out_degree"], dtype=np.int64)
    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), {"iterations": 0, "elapsed_seconds": 0.0, "converged": True}

    start = perf_counter()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    d_indptr = cuda.to_device(indptr)
    d_indices = cuda.to_device(indices)
    d_out_degree = cuda.to_device(out_degree)
    d_rank = cuda.to_device(rank)
    d_new_rank = cuda.device_array_like(d_rank)
    d_stats = cuda.to_device(np.zeros(3, dtype=np.float64))
    threads = 128
    blocks = (n + threads - 1) // threads
    converged = False
    l1_delta = float("inf")

    for iteration in range(1, max_iter + 1):
        reset_stats_kernel[1, 1](d_stats)
        dangling_mass_kernel[blocks, threads](d_out_degree, d_rank, d_stats)
        pull_update_kernel[blocks, threads](d_indptr, d_indices, d_out_degree, d_rank, d_new_rank, d_stats, alpha)
        stats = d_stats.copy_to_host()
        l1_delta = float(stats[1])
        rank_sum = float(stats[2])
        if rank_sum > 0.0:
            normalize_kernel[blocks, threads](d_new_rank, rank_sum)
        cuda.synchronize()
        d_rank, d_new_rank = d_new_rank, d_rank
        if l1_delta < tol:
            converged = True
            break

    rank = d_rank.copy_to_host()
    total = float(rank.sum())
    if total > 0:
        rank /= total

    elapsed = perf_counter() - start
    return rank, {
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "iterations": int(iteration),
        "elapsed_seconds": float(elapsed),
        "l1_delta": float(l1_delta),
        "rank_sum": float(rank.sum()),
        "converged": bool(converged),
        "mode": "gpu_v3_pull_gather",
        "fallback_note": "optimized pull_gather keeps rank vectors on device and copies only reductions per iteration.",
    }


def run(graph: dict, alpha: float, tol: float, max_iter: int, mode: str = "pull_gather") -> tuple[np.ndarray, dict]:
    require_cuda()
    if mode not in {"pull_gather", "push_scatter"}:
        raise ValueError(f"Unknown v3 mode: {mode}")
    if mode == "pull_gather":
        return _run_pull_gather(graph, alpha, tol, max_iter)

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
