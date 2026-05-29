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
    assert len(graph["indptr_out"]) == graph["num_nodes"] + 1
    assert len(graph["indices_out"]) == graph["num_edges"]


def test_outgoing_csr_rows_are_source_adjacency():
    graph = make_synthetic_graph("toy")
    indptr_out = graph["indptr_out"]
    indices_out = graph["indices_out"]
    assert indices_out[indptr_out[0] : indptr_out[1]].tolist() == [1, 2]
    assert indices_out[indptr_out[1] : indptr_out[2]].tolist() == [2]
    assert indices_out[indptr_out[2] : indptr_out[3]].tolist() == [0, 3]


def test_fixture_snap_sample_loads():
    graph = load_edge_list(Path("tests/fixtures/sample_snap.txt"))
    assert graph["num_nodes"] == 4
    assert graph["num_edges"] == 5
