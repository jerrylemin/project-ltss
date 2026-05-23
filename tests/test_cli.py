import subprocess
from pathlib import Path


def test_cpu_baseline_cli_config_runs(tmp_path: Path):
    repo_root = Path.cwd()
    result = subprocess.run(
        ["python", str(repo_root / "src/cpu_baseline.py"), "--config", str(repo_root / "project_spec.yaml")],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts/cpu_baseline_metrics.json").exists()
