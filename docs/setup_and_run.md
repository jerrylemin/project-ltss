# Setup and Run

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python src/cpu_baseline.py --config project_spec.yaml
python src/profile_cpu.py --config project_spec.yaml
python src/benchmark.py --config project_spec.yaml
```

CUDA is optional. GPU tests skip at module level when Numba cannot access CUDA.
