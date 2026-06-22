import numpy as np
import pytest
import scipy.sparse as sp
from numba import cuda

from src.data_loader import _graph_from_edges, make_synthetic_graph as _make_project_synthetic_graph
from src.gpu.pagerank_gpu import run_pagerank_gpu
from src.pagerank_cpu import run_pagerank_cpu


TOL = 1e-6


def make_synthetic_graph(n=200, seed=42):
    """Random directed graph: returns the project graph dict with incoming/outgoing CSR."""
    return _make_project_synthetic_graph("random", num_nodes=n, num_edges=n * 5, seed=seed, name="test_synthetic")


def make_dangling_graph():
    """Graph where node 0 has zero out-edges (dangling node)."""
    edges = [(1, 0), (2, 0), (2, 1)]
    return _graph_from_edges(edges, num_nodes=3, name="dangling", remap=False)


def scipy_pagerank(graph, alpha=0.85, tol=TOL, max_iter=200):
    """SciPy sparse reference -- ground truth."""
    indptr_in = np.asarray(graph["indptr"], dtype=np.int64)
    indices_in = np.asarray(graph["indices"], dtype=np.int64)
    out_degree = np.asarray(graph["out_degree"], dtype=np.int64)
    n = int(graph["num_nodes"])
    if n == 0:
        return np.array([], dtype=np.float64)

    data = 1.0 / np.maximum(out_degree[indices_in], 1).astype(np.float64)
    matrix = sp.csr_matrix((data, indices_in, indptr_in), shape=(n, n))
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    base = (1.0 - alpha) / n

    for _ in range(max_iter):
        dangling_mass = float(rank[out_degree == 0].sum())
        new_rank = base + alpha * (matrix @ rank + dangling_mass / n)
        total = float(new_rank.sum())
        if total > 0:
            new_rank /= total
        if float(np.abs(new_rank - rank).sum()) < tol:
            return np.asarray(new_rank, dtype=np.float64)
        rank = np.asarray(new_rank, dtype=np.float64)
    return rank


def scipy_pagerank_from_edges(edges, n, alpha=0.85, tol=TOL, max_iter=200):
    """Raw-edge SciPy reference that preserves self-loops and duplicate edges."""
    if not edges:
        return np.full(n, 1.0 / n, dtype=np.float64)

    src = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    dst = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    out_degree = np.bincount(src, minlength=n).astype(np.int64)
    data = 1.0 / out_degree[src].astype(np.float64)
    matrix = sp.csr_matrix((data, (dst, src)), shape=(n, n))
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    base = (1.0 - alpha) / n

    for _ in range(max_iter):
        dangling_mass = float(rank[out_degree == 0].sum())
        new_rank = base + alpha * (matrix @ rank + dangling_mass / n)
        total = float(new_rank.sum())
        if total > 0:
            new_rank /= total
        if float(np.abs(new_rank - rank).sum()) < tol:
            return np.asarray(new_rank, dtype=np.float64)
        rank = np.asarray(new_rank, dtype=np.float64)
    return rank


def numpy_pagerank(graph, tol=TOL):
    rank, _metrics = run_pagerank_cpu(graph, tol=tol, mode="numpy_loop", verify_scipy=False)
    return rank


def gpu_v1_pagerank(graph, tol=TOL):
    rank, _metrics = run_pagerank_gpu(graph, tol=tol, version="v1")
    return rank


def gpu_v2_pagerank(graph, tol=TOL):
    rank, _metrics = run_pagerank_gpu(graph, tol=tol, version="v2")
    return rank


def gpu_v3_pagerank(graph, mode="pull", tol=TOL):
    version = "v3_pull" if mode == "pull" else "v3_push"
    rank, _metrics = run_pagerank_gpu(graph, tol=tol, version=version)
    return rank


def relative_l1_error(actual, expected):
    return np.sum(np.abs(actual - expected)) / np.sum(np.abs(expected))


def assert_valid_rank(rank):
    assert not np.any(np.isnan(rank)), "NaN in rank vector"
    assert not np.any(np.isinf(rank)), "Inf in rank vector"
    assert not np.any(rank < -1e-12), "Negative rank value"
    assert abs(np.sum(rank) - 1.0) < 1e-5, "Rank vector does not sum to 1"


def test_cpu_vs_scipy_synthetic():
    """CPU NumPy CSR must match SciPy within 1e-6 L1 relative error."""
    graph = make_synthetic_graph()
    r_cpu = numpy_pagerank(graph, tol=TOL)
    r_scipy = scipy_pagerank(graph, tol=TOL)
    rel_err = relative_l1_error(r_cpu, r_scipy)
    assert rel_err < TOL, f"CPU vs SciPy L1 relative error {rel_err:.2e} >= 1e-6"


def test_cpu_dangling_node():
    """CPU must handle dangling nodes (zero out-degree) without NaN/Inf."""
    graph = make_dangling_graph()
    r = numpy_pagerank(graph, tol=TOL)
    assert not np.any(np.isnan(r)), "NaN in rank vector with dangling node"
    assert not np.any(np.isinf(r)), "Inf in rank vector with dangling node"
    assert abs(np.sum(r) - 1.0) < 1e-5, "Rank vector does not sum to 1"


def test_loader_preserves_self_loops_in_csr_structures():
    edges = [(0, 0), (0, 1), (1, 2), (2, 0)]
    graph = _graph_from_edges(edges, num_nodes=3, name="self_loop", remap=False)

    assert graph["num_edges"] == 4
    assert graph["out_degree"].tolist() == [2, 1, 1]
    assert graph["indices"][graph["indptr"][0] : graph["indptr"][1]].tolist() == [0, 2]
    assert graph["indices_out"][graph["indptr_out"][0] : graph["indptr_out"][1]].tolist() == [0, 1]


def test_self_loop_graph_matches_raw_edge_scipy_reference_cpu():
    edges = [(0, 0), (0, 1), (1, 2), (2, 0)]
    graph = _graph_from_edges(edges, num_nodes=3, name="self_loop", remap=False)
    r_cpu = numpy_pagerank(graph, tol=TOL)
    r_scipy = scipy_pagerank_from_edges(edges, 3)

    assert_valid_rank(r_cpu)
    assert relative_l1_error(r_cpu, r_scipy) < TOL


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("version", ["v1", "v2", "v3_pull", "v3_push"])
def test_self_loop_graph_matches_raw_edge_scipy_reference_gpu(version):
    edges = [(0, 0), (0, 1), (1, 2), (2, 0)]
    graph = _graph_from_edges(edges, num_nodes=3, name="self_loop", remap=False)
    r_gpu, _ = run_pagerank_gpu(graph, tol=TOL, version=version)
    r_scipy = scipy_pagerank_from_edges(edges, 3)

    assert_valid_rank(r_gpu)
    assert relative_l1_error(r_gpu, r_scipy) < TOL


def test_self_loop_with_dangling_node_matches_scipy_cpu():
    edges = [(0, 0), (1, 1), (2, 0), (2, 1)]
    graph = _graph_from_edges(edges, num_nodes=4, name="self_loop_dangling", remap=False)
    r_cpu = numpy_pagerank(graph, tol=TOL)
    r_scipy = scipy_pagerank_from_edges(edges, 4)

    assert_valid_rank(r_cpu)
    assert relative_l1_error(r_cpu, r_scipy) < TOL


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("version", ["v1", "v2", "v3_pull", "v3_push"])
def test_self_loop_with_dangling_node_matches_scipy_gpu(version):
    edges = [(0, 0), (1, 1), (2, 0), (2, 1)]
    graph = _graph_from_edges(edges, num_nodes=4, name="self_loop_dangling", remap=False)
    r_gpu, _ = run_pagerank_gpu(graph, tol=TOL, version=version)
    r_scipy = scipy_pagerank_from_edges(edges, 4)

    assert_valid_rank(r_gpu)
    assert relative_l1_error(r_gpu, r_scipy) < TOL


def test_single_node_self_loop_is_valid_cpu():
    edges = [(0, 0)]
    graph = _graph_from_edges(edges, num_nodes=1, name="single_self_loop", remap=False)
    r_cpu = numpy_pagerank(graph, tol=TOL)

    assert graph["num_edges"] == 1
    assert_valid_rank(r_cpu)
    assert r_cpu[0] == pytest.approx(1.0)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_gpu_v1_vs_scipy():
    """GPU V1 output must match SciPy within 1e-6 L1 relative error."""
    graph = make_synthetic_graph()
    r_v1 = gpu_v1_pagerank(graph, tol=TOL)
    r_scipy = scipy_pagerank(graph, tol=TOL)
    rel_err = relative_l1_error(r_v1, r_scipy)
    assert rel_err < TOL, f"V1 vs SciPy error {rel_err:.2e} >= 1e-6"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_gpu_v2_vs_v1():
    """GPU V2 must produce identical output to V1 within 1e-6."""
    graph = make_synthetic_graph()
    r_v1 = gpu_v1_pagerank(graph, tol=TOL)
    r_v2 = gpu_v2_pagerank(graph, tol=TOL)
    rel_err = relative_l1_error(r_v2, r_v1)
    assert rel_err < TOL, f"V2 vs V1 error {rel_err:.2e} >= 1e-6"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_gpu_v3_push_vs_pull():
    """V3 push and pull must produce identical output within 1e-6."""
    graph = make_synthetic_graph()
    r_pull = gpu_v3_pagerank(graph, mode="pull", tol=TOL)
    r_push = gpu_v3_pagerank(graph, mode="push", tol=TOL)
    rel_err = relative_l1_error(r_push, r_pull)
    assert rel_err < TOL, f"V3 push vs pull error {rel_err:.2e} >= 1e-6"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("mode", ["pull", "push"])
def test_gpu_v3_vs_scipy(mode):
    graph = make_synthetic_graph()
    r_v3 = gpu_v3_pagerank(graph, mode=mode, tol=TOL)
    r_scipy = scipy_pagerank(graph, tol=TOL)
    rel_err = relative_l1_error(r_v3, r_scipy)
    assert rel_err < TOL, f"V3-{mode} vs SciPy error {rel_err:.2e} >= 1e-6"
