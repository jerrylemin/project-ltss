# Setup And Run

Last updated: 2026-05-29T14:03:47+07:00

Create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the core checks:

```powershell
.\.venv\Scripts\python.exe src\cpu_baseline.py --graph data\graphs\roadNet-CA.tsv
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe src\benchmark.py
```

Download the five required SNAP graphs locally:

```powershell
.\.venv\Scripts\python.exe scripts/download_graphs.py
```

Downloaded graph TSVs are written to `data/graphs/` and ignored by git.

CUDA verification:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "from numba import cuda; print(cuda.is_available()); print(cuda.detect())"
```

Final artifact checks:

```powershell
Select-String -Path artifacts\benchmark_results.csv -Pattern "com-youtube"
Select-String -Path src\gpu\*.py -Pattern "shfl_down_sync"
Select-String -Path src\gpu\pagerank_v1.py -Pattern "copy_to_host|to_device"
```
