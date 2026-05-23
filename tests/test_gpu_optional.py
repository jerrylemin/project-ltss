import numpy as np
import pytest

cuda = pytest.importorskip("numba.cuda")

if not cuda.is_available():
    pytest.skip("CUDA is not available; GPU tests skipped cleanly.", allow_module_level=True)

from src.data_loader import make_synthetic_graph
from src.gpu.pagerank_gpu import run_pagerank_gpu
from src.metrics import relative_l1_error
from src.pagerank_cpu import run_pagerank_cpu


def test_gpu_v1_matches_cpu_when_cuda_available():
    graph = make_synthetic_graph("toy")
    cpu_rank, _ = run_pagerank_cpu(graph)
    gpu_rank, metrics = run_pagerank_gpu(graph, version="v1")
    assert metrics["rank_sum"] == pytest.approx(1.0, abs=1e-4)
    assert np.all(gpu_rank >= 0)
    assert relative_l1_error(gpu_rank, cpu_rank) < 1e-4
