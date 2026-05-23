from pathlib import Path

from src.config import load_config


def test_project_spec_required_fields_exist():
    assert Path("project_spec.yaml").exists()
    config = load_config("project_spec.yaml")
    assert config["project_id"] == "C3"
    assert config["title"] == "PageRank"
    assert "alpha" in config["algorithm"]
    assert "tolerance" in config["algorithm"]
