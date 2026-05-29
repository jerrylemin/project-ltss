"""
pagerank_v1.py - GPU V1: one-thread-per-row CSR PageRank

Strategy   : one CUDA thread per destination row gathers incoming CSR edges.
Key opt    : replaces the CPU CSR SpMV loop with a custom CUDA SpMV kernel.
Expected   : exposes PageRank edge traversal parallelism while keeping damping separate.
Limitation : one thread serially scans high-degree rows, so load imbalance remains.
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

import numpy as np

from src.gpu.cuda_utils import require_cuda


def _device_scalar_to_float(device_scalar) -> float:
    return float(device_scalar.copy_to_host()[0])


def cuda_available() -> bool:
    from src.gpu.cuda_utils import cuda_available as _cuda_available

    return _cuda_available()


def run(graph: dict, alpha: float, tol: float, max_iter: int) -> tuple[np.ndarray, dict]:
    require_cuda()
    from numba import cuda

    @cuda.jit
    def reset_scalar_kernel(value):
        """Clear one scalar device buffer."""
        if cuda.grid(1) == 0:
            value[0] = 0.0

    @cuda.jit
    def dangling_mass_kernel(out_degree, rank, dangling_mass):
        """One thread checks one node and atomically accumulates dangling rank mass."""
        node = cuda.grid(1)
        if node < out_degree.size and out_degree[node] == 0:
            cuda.atomic.add(dangling_mass, 0, rank[node])

    @cuda.jit
    def csr_spmv_kernel(indptr, indices, out_degree, rank, partial):
        """
        V1: one thread per row CSR SpMV.
        Bell & Garland [3] motivate CSR row-parallel SpMV on throughput-oriented GPUs.
        """
        node = cuda.grid(1)
        n = out_degree.size
        if node < n:
            total = 0.0
            for offset in range(indptr[node], indptr[node + 1]):
                src = indices[offset]
                degree = out_degree[src]
                if degree > 0:
                    total += rank[src] / degree
            partial[node] = total

    @cuda.jit
    def damping_update_kernel(partial, out_degree, old_rank, new_rank, alpha_value, dangling_mass, l1_delta):
        """
        Separate damping kernel for PageRank update.
        Page et al. [1] define the damped random-surfer PageRank iteration.
        """
        node = cuda.grid(1)
        n = old_rank.size
        if node < n:
            value = (1.0 - alpha_value) / n + alpha_value * (partial[node] + dangling_mass[0] / n)
            new_rank[node] = value
            cuda.atomic.add(l1_delta, 0, abs(value - old_rank[node]))

    indptr = np.asarray(graph["indptr"])
    indices = np.asarray(graph["indices"])
    out_degree = np.asarray(graph["out_degree"])
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
    d_partial = cuda.device_array(n, dtype=np.float64)
    d_dangling = cuda.to_device(np.zeros(1, dtype=np.float64))
    d_delta = cuda.to_device(np.zeros(1, dtype=np.float64))
    threads = 128
    blocks = (n + threads - 1) // threads
    converged = False
    l1_delta = float("inf")

    for iteration in range(1, max_iter + 1):
        reset_scalar_kernel[1, 1](d_dangling)
        reset_scalar_kernel[1, 1](d_delta)
        dangling_mass_kernel[blocks, threads](d_out_degree, d_rank, d_dangling)
        csr_spmv_kernel[blocks, threads](d_indptr, d_indices, d_out_degree, d_rank, d_partial)
        damping_update_kernel[blocks, threads](d_partial, d_out_degree, d_rank, d_new_rank, alpha, d_dangling, d_delta)
        cuda.synchronize()
        l1_delta = _device_scalar_to_float(d_delta)
        if l1_delta < tol:
            converged = True
            d_rank = d_new_rank
            break
        d_rank, d_new_rank = d_new_rank, d_rank

    elapsed = perf_counter() - start
    rank = d_rank.copy_to_host()
    total = float(rank.sum())
    if total > 0:
        rank /= total
    return rank, {
        "num_nodes": int(graph["num_nodes"]),
        "num_edges": int(graph["num_edges"]),
        "iterations": int(iteration),
        "elapsed_seconds": float(elapsed),
        "l1_delta": float(l1_delta),
        "rank_sum": float(rank.sum()),
        "converged": bool(converged),
        "mode": "gpu_v1",
    }
