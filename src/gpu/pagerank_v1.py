from __future__ import annotations

from time import perf_counter

import numpy as np

from src.gpu.cuda_utils import require_cuda


def cuda_available() -> bool:
    from src.gpu.cuda_utils import cuda_available as _cuda_available

    return _cuda_available()


def run(graph: dict, alpha: float, tol: float, max_iter: int) -> tuple[np.ndarray, dict]:
    require_cuda()
    from numba import cuda

    @cuda.jit
    def csr_spmv_kernel(indptr, indices, out_degree, rank, partial):
        """One CUDA thread owns one destination node and gathers incoming CSR contributions."""
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
        """One CUDA thread applies damping to one node and atomically accumulates the L1 delta."""
        node = cuda.grid(1)
        n = old_rank.size
        if node < n:
            value = (1.0 - alpha_value) / n + alpha_value * (partial[node] + dangling_mass / n)
            new_rank[node] = value
            cuda.atomic.add(l1_delta, 0, abs(value - old_rank[node]))

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
    d_partial = cuda.device_array(n, dtype=np.float64)
    threads = 128
    blocks = (n + threads - 1) // threads
    converged = False
    l1_delta = float("inf")

    for iteration in range(1, max_iter + 1):
        dangling_mass = float(rank[out_degree == 0].sum())
        d_delta = cuda.to_device(np.array([0.0], dtype=np.float64))
        csr_spmv_kernel[blocks, threads](d_indptr, d_indices, d_out_degree, d_rank, d_partial)
        damping_update_kernel[blocks, threads](d_partial, d_out_degree, d_rank, d_new_rank, alpha, dangling_mass, d_delta)
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
        "mode": "gpu_v1",
    }
