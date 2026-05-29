from pathlib import Path

from src.ui.dashboard_server import detect_graph_files, get_project_status, load_benchmark_csv


def test_load_benchmark_csv_missing(tmp_path: Path):
    """Returns status=missing when file does not exist."""
    result = load_benchmark_csv(str(tmp_path / "missing.csv"))
    assert result == {"status": "missing", "rows": []}


def test_load_benchmark_csv_valid(tmp_path: Path):
    """Parses a valid CSV and returns correct row count."""
    csv_path = tmp_path / "benchmark.csv"
    csv_path.write_text("graph_name,version,convergence_time_s\nroadNet-CA,cpu_numpy,1.0\ncom-youtube,gpu_v3_push,0.1\n")

    result = load_benchmark_csv(str(csv_path))

    assert result["status"] == "ok"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["graph_name"] == "roadNet-CA"


def test_load_benchmark_csv_malformed(tmp_path: Path):
    """Returns status=error for a malformed CSV without raising."""
    csv_path = tmp_path / "benchmark.csv"
    csv_path.write_text("graph_name,version\nroadNet-CA\n")

    result = load_benchmark_csv(str(csv_path))

    assert result["status"] == "error"
    assert "error" in result


def test_detect_graph_files_missing_dir(tmp_path: Path):
    """Returns status=missing when directory does not exist."""
    result = detect_graph_files(str(tmp_path / "graphs"))
    assert result == {"status": "missing", "files": []}


def test_detect_graph_files_populated(tmp_path: Path):
    """Returns correct list of .tsv files in directory."""
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    (graph_dir / "b.tsv").write_text("1\t2\n")
    (graph_dir / "a.tsv").write_text("2\t3\n")
    (graph_dir / "ignore.txt").write_text("not a graph\n")

    result = detect_graph_files(str(graph_dir))

    assert result == {"status": "ok", "files": ["a.tsv", "b.tsv"]}


def test_get_project_status_structure():
    """Return value has required keys."""
    result = get_project_status()

    assert "benchmark_csv" in result
    assert "graph_files" in result
    assert "source_checks" in result
    assert "checklist" in result
