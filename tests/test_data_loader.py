from pathlib import Path

from src.data_loader import load_edge_list, make_synthetic_graph


def test_read_edge_list_with_comments(tmp_path: Path):
    edge_file = tmp_path / "sample.edges"
    edge_file.write_text("# comment\n\n10 20\n20 30\n", encoding="utf-8")
    graph = load_edge_list(edge_file)
    assert graph["num_nodes"] == 3
    assert graph["num_edges"] == 2


def test_auto_remap_non_contiguous_node_ids(tmp_path: Path):
    edge_file = tmp_path / "sample.txt"
    edge_file.write_text("100 500\n500 900\n", encoding="utf-8")
    graph = load_edge_list(edge_file)
    assert graph["num_nodes"] == 3
    assert max(graph["indices"].tolist()) <= 2


def test_synthetic_toy_graph_is_valid():
    graph = make_synthetic_graph("toy")
    assert graph["name"] == "toy"
    assert graph["num_nodes"] == 4
    assert graph["num_edges"] > 0
    assert len(graph["indptr"]) == graph["num_nodes"] + 1
