from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np


Graph = dict[str, object]


def _empty_graph(n: int, name: str) -> Graph:
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.array([], dtype=np.int32)
    out_degree = np.zeros(n, dtype=np.int32)
    return {
        "indptr": indptr,
        "indices": indices,
        "indptr_in": indptr,
        "indices_in": indices,
        "indptr_out": indptr.copy(),
        "indices_out": indices.copy(),
        "out_degree": out_degree,
        "num_nodes": n,
        "num_edges": 0,
        "name": name,
    }


def _graph_from_arrays(src: np.ndarray, dst: np.ndarray, *, num_nodes: int | None = None, name: str = "graph") -> Graph:
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    if src.size == 0:
        return _empty_graph(int(num_nodes or 0), name)

    n = int(num_nodes) if num_nodes is not None else int(max(int(src.max()), int(dst.max())) + 1)
    valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
    if not np.all(valid):
        raise ValueError("Edge list contains node ids outside the declared node range")

    order = np.lexsort((src, dst))
    src_sorted = src[order].astype(np.int32, copy=False)
    dst_sorted = dst[order].astype(np.int32, copy=False)
    counts = np.bincount(dst_sorted, minlength=n).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    out_order = np.lexsort((dst, src))
    src_out_sorted = src[out_order].astype(np.int32, copy=False)
    dst_out_sorted = dst[out_order].astype(np.int32, copy=False)
    out_counts = np.bincount(src_out_sorted, minlength=n).astype(np.int64)
    indptr_out = np.zeros(n + 1, dtype=np.int64)
    indptr_out[1:] = np.cumsum(out_counts)
    out_degree = np.bincount(src, minlength=n).astype(np.int32)

    return {
        "indptr": indptr,
        "indices": src_sorted,
        "indptr_in": indptr,
        "indices_in": src_sorted,
        "indptr_out": indptr_out,
        "indices_out": dst_out_sorted,
        "out_degree": out_degree,
        "num_nodes": int(n),
        "num_edges": int(len(src_sorted)),
        "name": name,
    }


def _graph_from_edges(
    edges: Iterable[tuple[int, int]],
    *,
    num_nodes: int | None = None,
    name: str = "graph",
    remap: bool = True,
) -> Graph:
    raw_edges = list(edges)
    if not raw_edges:
        n = int(num_nodes or 0)
        return _empty_graph(n, name)

    mapped_source_edges = [(int(src), int(dst)) for src, dst in raw_edges]

    if num_nodes is not None:
        n = int(num_nodes)
        mapped_edges = mapped_source_edges
    elif remap:
        node_ids = sorted({node for edge in mapped_source_edges for node in edge})
        mapping = {node_id: idx for idx, node_id in enumerate(node_ids)}
        mapped_edges = [(mapping[src], mapping[dst]) for src, dst in mapped_source_edges]
        n = len(node_ids)
    else:
        max_node = max(max(src, dst) for src, dst in mapped_source_edges)
        n = max_node + 1
        mapped_edges = mapped_source_edges

    if not mapped_edges:
        return _empty_graph(n, name)

    src = np.asarray([edge[0] for edge in mapped_edges], dtype=np.int32)
    dst = np.asarray([edge[1] for edge in mapped_edges], dtype=np.int32)
    return _graph_from_arrays(src, dst, num_nodes=n, name=name)


def load_edge_list(path: str | Path, *, remap: bool = True) -> Graph:
    graph, _timings = load_edge_list_with_timings(path, remap=remap)
    return graph


def load_edge_list_with_timings(path: str | Path, *, remap: bool = True) -> tuple[Graph, dict[str, float]]:
    edge_path = Path(path)
    start = perf_counter()
    if not remap:
        read_start = perf_counter()
        edges = np.loadtxt(edge_path, dtype=np.int32, comments="#", ndmin=2, usecols=(0, 1))
        read_time = perf_counter() - read_start
        if edges.size == 0:
            graph = _empty_graph(0, edge_path.stem)
            return graph, {
                "load_time_seconds": perf_counter() - start,
                "file_read_time_seconds": read_time,
                "csr_build_time_seconds": 0.0,
            }
        csr_start = perf_counter()
        graph = _graph_from_arrays(edges[:, 0], edges[:, 1], name=edge_path.stem)
        csr_time = perf_counter() - csr_start
        return graph, {
            "load_time_seconds": perf_counter() - start,
            "file_read_time_seconds": read_time,
            "csr_build_time_seconds": csr_time,
        }

    edges: list[tuple[int, int]] = []
    read_start = perf_counter()
    with edge_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                edges.append((int(parts[0]), int(parts[1])))
            except ValueError:
                if line_number == 1:
                    continue
                raise
    read_time = perf_counter() - read_start
    csr_start = perf_counter()
    graph = _graph_from_edges(edges, name=edge_path.stem, remap=True)
    csr_time = perf_counter() - csr_start
    return graph, {
        "load_time_seconds": perf_counter() - start,
        "file_read_time_seconds": read_time,
        "csr_build_time_seconds": csr_time,
    }


def make_synthetic_graph(
    graph_type: str = "random",
    *,
    num_nodes: int = 1000,
    num_edges: int = 5000,
    seed: int = 42,
    name: str | None = None,
) -> Graph:
    graph_type = graph_type.lower()
    graph_name = name or f"synthetic_{graph_type}_{num_nodes}_{num_edges}"

    if graph_type == "toy":
        toy_edges = [(0, 1), (0, 2), (1, 2), (2, 0), (2, 3), (3, 0)]
        return _graph_from_edges(toy_edges, num_nodes=4, name=name or "toy", remap=False)

    if graph_type == "chain":
        edges = [(i, i + 1) for i in range(max(0, num_nodes - 1))]
        return _graph_from_edges(edges, num_nodes=num_nodes, name=graph_name, remap=False)

    if graph_type == "star":
        edges = [(0, i) for i in range(1, num_nodes)]
        edges += [(i, 0) for i in range(1, num_nodes)]
        return _graph_from_edges(edges[:num_edges], num_nodes=num_nodes, name=graph_name, remap=False)

    rng = np.random.default_rng(seed)
    if graph_type == "random":
        src = rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)
        dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)
    elif graph_type == "power_law":
        src = np.minimum(rng.zipf(2.0, size=num_edges) - 1, num_nodes - 1).astype(np.int64)
        dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)
        src = (src + rng.integers(0, num_nodes, size=num_edges, dtype=np.int64)) % num_nodes
    else:
        raise ValueError(f"Unknown synthetic graph type: {graph_type}")

    if num_nodes > 1:
        self_loop = src == dst
        dst[self_loop] = (dst[self_loop] + 1) % num_nodes

    edges = zip(src.tolist(), dst.tolist())
    return _graph_from_edges(edges, num_nodes=num_nodes, name=graph_name, remap=False)


def load_graph_from_config(config: dict, graph_size: str = "small") -> Graph:
    dataset = config.get("dataset", {})
    edges_path = dataset.get("edges_path")
    if edges_path:
        return load_edge_list(edges_path)
    sizes = dataset.get("synthetic_sizes", {})
    spec = sizes.get(graph_size, sizes.get("small", {"nodes": 1000, "edges": 5000}))
    return make_synthetic_graph(
        "random",
        num_nodes=int(spec["nodes"]),
        num_edges=int(spec["edges"]),
        name=f"synthetic_{graph_size}",
    )
