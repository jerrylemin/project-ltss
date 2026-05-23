import numpy as np
import pytest

from src.data_loader import make_synthetic_graph
from src.metrics import relative_l1_error
from src.pagerank_cpu import run_pagerank_cpu


def test_rank_sum_and_non_negative_values():
    graph = make_synthetic_graph("toy")
    rank, metrics = run_pagerank_cpu(graph)
    assert 0.9999 <= metrics["rank_sum"] <= 1.0001
    assert np.all(rank >= 0)


def test_cpu_vs_scipy_relative_error():
    pytest.importorskip("scipy")
    graph = make_synthetic_graph("toy")
    rank, _ = run_pagerank_cpu(graph, mode="numpy_loop")
    ref_rank, _ = run_pagerank_cpu(graph, mode="scipy", verify_scipy=False)
    assert relative_l1_error(rank, ref_rank) < 1e-5


def test_toy_graph_deterministic_across_runs():
    graph1 = make_synthetic_graph("toy")
    graph2 = make_synthetic_graph("toy")
    rank1, _ = run_pagerank_cpu(graph1)
    rank2, _ = run_pagerank_cpu(graph2)
    np.testing.assert_allclose(rank1, rank2)
