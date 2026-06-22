"""
pagerank_v3.py - GPU V3: warp-shuffle push vs pull PageRank

Strategy   : compare pull gather and source-owned push scatter PageRank kernels.
Key opt    : warp-shuffle convergence reductions and device-resident rank vectors.
Expected   : reduces reduction overhead and exposes push/pull behavior by graph type.
Limitation : push can suffer load imbalance and atomic scatter contention on hubs.
"""

# References:
# [1] Page et al. (1999). The PageRank Citation Ranking: Bringing Order to
#     the Web. Stanford TR. https://infolab.stanford.edu/pub/papers/google.pdf
# [2] McLaughlin & Bader (2014). Scalable and High Performance PageRank on
#     the GPU. SC 2014. Push/pull strategy, warp-level reduction.
# [3] Bell & Garland (2009). Implementing SpMV on Throughput-Oriented
#     Processors. SC 2009. CSR SpMV optimization patterns.

from __future__ import annotations

from time import perf_counter
from typing import Callable

import numpy as np

from src.gpu.cuda_utils import require_cuda

_PULL_KERNELS: tuple[Callable, ...] | None = None
_PUSH_KERNELS: tuple[Callable, ...] | None = None


def cuda_available() -> bool:
    from src.gpu.cuda_utils import cuda_available as _cuda_available

    return _cuda_available()


def _outgoing_csr_from_graph(graph: dict) -> tuple[np.ndarray, np.ndarray]:
    if "indptr_out" in graph and "indices_out" in graph:
        return (
            np.asarray(graph["indptr_out"]),
            np.asarray(graph["indices_out"]),
        )

    indptr = np.asarray(graph["indptr"])
    indices = np.asarray(graph["indices"])
    n = int(graph["num_nodes"])
    dst = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    src = indices.copy()
    order = np.lexsort((dst, src))
    src_sorted = src[order]
    dst_sorted = dst[order]
    counts = np.bincount(src_sorted, minlength=n).astype(np.int64)
    indptr_out = np.zeros(n + 1, dtype=np.int64)
    indptr_out[1:] = np.cumsum(counts)
    return indptr_out, dst_sorted.astype(np.int64, copy=False)


def _pull_kernels() -> tuple[Callable, ...]:
    global _PULL_KERNELS
    if _PULL_KERNELS is not None:
        return _PULL_KERNELS

    from numba import cuda

    @cuda.jit
    def reset_scalar_kernel(value):
        """Clear one scalar device buffer."""
        if cuda.grid(1) == 0:
            value[0] = 0.0

    @cuda.jit
    def dangling_mass_kernel(out_degree, rank, dangling_mass):
        """One thread checks one node and atomically accumulates rank mass for dangling nodes."""
        node = cuda.grid(1)
        if node < out_degree.size and out_degree[node] == 0:
            cuda.atomic.add(dangling_mass, 0, rank[node])

    # V3-pull: pull-based (gather) PageRank with warp-shuffle L1 reduction
    # McLaughlin & Bader (2014), Scalable and High Performance PageRank on GPU
    @cuda.jit
    def pull_update_kernel(indptr, indices, out_degree, old_rank, new_rank, dangling_mass, conv_d, alpha_value):
        """
        Pull-based (gather) PageRank with warp-shuffle L1 reduction.
        McLaughlin & Bader [2] discuss pull as avoiding atomic scatter contention.
        Warp shuffle pattern follows the reduction style motivated by Bell & Garland [3].
        """
        src = cuda.grid(1)
        n = old_rank.size
        if src < n:
            total = 0.0
            for offset in range(indptr[src], indptr[src + 1]):
                incoming_src = indices[offset]
                degree = out_degree[incoming_src]
                if degree > 0:
                    total += old_rank[incoming_src] / degree
            value = (1.0 - alpha_value) / n + alpha_value * (total + dangling_mass[0] / n)
            new_rank[src] = value
            diff = abs(value - old_rank[src])
            lane = src & 31
            diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 16)
            diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 8)
            diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 4)
            diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 2)
            diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 1)
            if lane == 0:
                cuda.atomic.add(conv_d, 0, diff)

    _PULL_KERNELS = (reset_scalar_kernel, dangling_mass_kernel, pull_update_kernel)
    return _PULL_KERNELS


def _push_kernels() -> tuple[Callable, ...]:
    global _PUSH_KERNELS
    if _PUSH_KERNELS is not None:
        return _PUSH_KERNELS

    from numba import cuda

    @cuda.jit
    def reset_scalar_kernel(value):
        """Clear one scalar device buffer."""
        if cuda.grid(1) == 0:
            value[0] = 0.0

    @cuda.jit
    def dangling_mass_kernel(out_degree, rank, dangling_mass):
        """One thread checks one node and atomically accumulates dangling rank mass."""
        src = cuda.grid(1)
        if src < out_degree.size and out_degree[src] == 0:
            cuda.atomic.add(dangling_mass, 0, rank[src])

    @cuda.jit
    def initialize_rank_kernel(rank_out, dangling_mass, damping, n_nodes):
        """Set rank_out to teleportation plus dangling-mass contribution before scatter."""
        src = cuda.grid(1)
        if src < rank_out.size:
            rank_out[src] = (1.0 - damping) / n_nodes + damping * dangling_mass[0] / n_nodes

    # V3-push: push-based (scatter) PageRank with warp-shuffle L1 reduction
    # McLaughlin & Bader (2014), Scalable and High Performance PageRank on GPU
    @cuda.jit
    def pagerank_push_v3(indptr_out, indices_out, rank_in, rank_out, out_degree, damping, n_nodes):
        """
        V3 push-based (scatter) PageRank.
        McLaughlin & Bader [2] -- Algorithm 1 (push variant).
        Each thread = one SOURCE node; scatters rank to all destinations.
        Convergence is measured after scatter by exact_l1_kernel so all atomic
        contributions are visible before the L1 delta is reduced.
        """
        src = cuda.grid(1)
        if src >= n_nodes:
            return

        contrib = 0.0
        if out_degree[src] > 0:
            contrib = rank_in[src] / out_degree[src]
            for j in range(indptr_out[src], indptr_out[src + 1]):
                dst = indices_out[j]
                cuda.atomic.add(rank_out, dst, damping * contrib)

    @cuda.jit
    def exact_l1_kernel(rank_in, rank_out, conv_d):
        """Compute exact post-scatter L1 convergence with warp-shuffle reduction."""
        src = cuda.grid(1)
        if src >= rank_in.size:
            return

        diff = abs(rank_in[src] - rank_out[src])
        lane = src & 31
        diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 16)
        diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 8)
        diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 4)
        diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 2)
        diff += cuda.shfl_down_sync(0xFFFFFFFF, diff, 1)
        if lane == 0:
            cuda.atomic.add(conv_d, 0, diff)

    _PUSH_KERNELS = (
        reset_scalar_kernel,
        dangling_mass_kernel,
        initialize_rank_kernel,
        pagerank_push_v3,
        exact_l1_kernel,
    )
    return _PUSH_KERNELS


def _run_pull_gather(graph: dict, alpha: float, tol: float, max_iter: int) -> tuple[np.ndarray, dict]:
    require_cuda()
    from numba import cuda

    reset_scalar_kernel, dangling_mass_kernel, pull_update_kernel = _pull_kernels()

    indptr = np.asarray(graph["indptr"])
    indices = np.asarray(graph["indices"])
    out_degree = np.asarray(graph["out_degree"])
    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), {"iterations": 0, "elapsed_seconds": 0.0, "converged": True}

    total_start = perf_counter()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    h2d_start = perf_counter()
    d_indptr = cuda.to_device(indptr)
    d_indices = cuda.to_device(indices)
    d_out_degree = cuda.to_device(out_degree)
    d_rank = cuda.to_device(rank)
    d_new_rank = cuda.device_array_like(d_rank)
    d_dangling = cuda.to_device(np.zeros(1, dtype=np.float64))
    d_delta = cuda.to_device(np.zeros(1, dtype=np.float64))
    cuda.synchronize()
    h2d_time = perf_counter() - h2d_start
    threads = 128
    blocks = (n + threads - 1) // threads
    converged = False
    l1_delta = float("inf")

    kernel_start = perf_counter()
    for iteration in range(1, max_iter + 1):
        reset_scalar_kernel[1, 1](d_dangling)
        reset_scalar_kernel[1, 1](d_delta)
        dangling_mass_kernel[blocks, threads](d_out_degree, d_rank, d_dangling)
        pull_update_kernel[blocks, threads](d_indptr, d_indices, d_out_degree, d_rank, d_new_rank, d_dangling, d_delta, alpha)
        cuda.synchronize()
        l1_delta = float(d_delta.copy_to_host()[0])
        d_rank, d_new_rank = d_new_rank, d_rank
        if l1_delta < tol:
            converged = True
            break

    cuda.synchronize()
    kernel_time = perf_counter() - kernel_start
    d2h_start = perf_counter()
    rank = d_rank.copy_to_host()
    d2h_time = perf_counter() - d2h_start
    total = float(rank.sum())
    if total > 0:
        rank /= total

    elapsed = perf_counter() - total_start
    return rank, {
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "iterations": int(iteration),
        "elapsed_seconds": float(elapsed),
        "h2d_time_seconds": float(h2d_time),
        "kernel_time_seconds": float(kernel_time),
        "d2h_time_seconds": float(d2h_time),
        "convergence_time_seconds": float(kernel_time),
        "total_time_seconds": float(elapsed),
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

    (
        reset_scalar_kernel,
        dangling_mass_kernel,
        initialize_rank_kernel,
        pagerank_push_v3,
        exact_l1_kernel,
    ) = _push_kernels()

    out_degree = np.asarray(graph["out_degree"])
    n = len(out_degree)
    if n == 0:
        return np.array([], dtype=np.float64), {"iterations": 0, "elapsed_seconds": 0.0, "converged": True}
    indptr_out, indices_out = _outgoing_csr_from_graph(graph)

    total_start = perf_counter()
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    h2d_start = perf_counter()
    d_indptr_out = cuda.to_device(indptr_out)
    d_indices_out = cuda.to_device(indices_out)
    d_out_degree = cuda.to_device(out_degree)
    d_rank = cuda.to_device(rank)
    d_new_rank = cuda.device_array_like(d_rank)
    d_dangling = cuda.to_device(np.zeros(1, dtype=np.float64))
    d_delta = cuda.to_device(np.zeros(1, dtype=np.float64))
    cuda.synchronize()
    h2d_time = perf_counter() - h2d_start
    threads = 128
    node_blocks = (n + threads - 1) // threads
    converged = False
    l1_delta = float("inf")

    kernel_start = perf_counter()
    for iteration in range(1, max_iter + 1):
        reset_scalar_kernel[1, 1](d_dangling)
        reset_scalar_kernel[1, 1](d_delta)
        dangling_mass_kernel[node_blocks, threads](d_out_degree, d_rank, d_dangling)
        initialize_rank_kernel[node_blocks, threads](d_new_rank, d_dangling, alpha, n)
        pagerank_push_v3[
            node_blocks,
            threads,
        ](d_indptr_out, d_indices_out, d_rank, d_new_rank, d_out_degree, alpha, n)
        exact_l1_kernel[node_blocks, threads](d_rank, d_new_rank, d_delta)
        cuda.synchronize()
        l1_delta = float(d_delta.copy_to_host()[0])
        if l1_delta < tol:
            converged = True
            d_rank = d_new_rank
            break
        d_rank, d_new_rank = d_new_rank, d_rank

    cuda.synchronize()
    kernel_time = perf_counter() - kernel_start
    d2h_start = perf_counter()
    rank = d_rank.copy_to_host()
    d2h_time = perf_counter() - d2h_start
    total = float(rank.sum())
    if total > 0:
        rank /= total
    elapsed = perf_counter() - total_start
    return rank, {
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "iterations": int(iteration),
        "elapsed_seconds": float(elapsed),
        "h2d_time_seconds": float(h2d_time),
        "kernel_time_seconds": float(kernel_time),
        "d2h_time_seconds": float(d2h_time),
        "convergence_time_seconds": float(kernel_time),
        "total_time_seconds": float(elapsed),
        "l1_delta": float(l1_delta),
        "rank_sum": float(rank.sum()),
        "converged": bool(converged),
        "mode": "gpu_v3_push_scatter",
        "fallback_note": "source-owned push scatter with warp-shuffle L1 reduction.",
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
