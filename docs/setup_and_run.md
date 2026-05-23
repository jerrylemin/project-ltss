# Setup and Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe src/cpu_baseline.py --config project_spec.yaml
.\.venv\Scripts\python.exe src/profile_cpu.py --config project_spec.yaml
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml
powershell -ExecutionPolicy Bypass -File scripts/check_cuda.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_gpu_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_gpu_benchmark.ps1
```

CUDA is optional. GPU tests skip at module level when Numba cannot access CUDA.

SNAP sample:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_snap_sample.ps1
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml --edges-path data/raw/sample_snap.txt
```
