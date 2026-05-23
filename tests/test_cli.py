import subprocess


def test_cpu_baseline_cli_config_runs():
    result = subprocess.run(
        ["python", "src/cpu_baseline.py", "--config", "project_spec.yaml"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
