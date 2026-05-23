# LTSS Project 1 - C3 PageRank

## 1. Ten project va Topic C3 PageRank

Project LTSS trien khai PageRank cho topic C3, track Graph and Sparse. Repo muc tieu: `https://github.com/jerrylemin/project-ltss`.

Team:

- Le Minh, Team Leader, integration, docs, GitHub.
- Nguyen Vu Bach, GPU kernels, benchmark, testing.

## 2. Muc tieu va Partial GPU Principle

Muc tieu la do CPU baseline PageRank tren do thi sparse, xac dinh bottleneck SpMV trong vong lap PageRank, roi tang toc phan kernel lap bang GPU. Cac buoc doc du lieu, cau hinh, validate, bao cao va fallback van chay tren CPU de dam bao chuong trinh dung duoc tren may khong co GPU.

## 3. Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Neu PowerShell chan activate script tren UNC path, dung truc tiep `.\.venv\Scripts\python.exe`.

## 4. Chay CPU baseline / profile / benchmark / tests

```powershell
.\.venv\Scripts\python.exe src/cpu_baseline.py --config project_spec.yaml
.\.venv\Scripts\python.exe src/profile_cpu.py --config project_spec.yaml
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml --gpu
.\.venv\Scripts\python.exe -m pytest -q
```

## 5. Dataset: SNAP va synthetic fallback

Mac dinh project dung synthetic graph trong `project_spec.yaml`, nen khong can dataset that de chay. Neu co SNAP edge list local, dat duong dan vao `dataset.edges_path` hoac chay:

```powershell
.\.venv\Scripts\python.exe src/cpu_baseline.py --edges data/raw/com-youtube.ungraph.txt
.\.venv\Scripts\python.exe src/benchmark.py --config project_spec.yaml --edges-path data/raw/com-youtube.ungraph.txt
```

Tao sample nho:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_snap_sample.ps1
```

## 6. Cau truc repo

```text
src/        code CPU, profile, benchmark, GPU optional
tests/      pytest suite
docs/       proposal, benchmark report, handoff, checklist
artifacts/  metrics JSON, environment check, benchmark CSV
scripts/    helper PowerShell scripts
```

## 7. Trang thai GPU (auto-detect)

GPU duoc detect bang Numba CUDA. Moi truong hien tai da cai CUDA Toolkit 13.2 va `numba-cuda==0.30.2`; `cuda.is_available()` tra ve `True` khi script nap CUDA Toolkit PATH.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_cuda.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_gpu_smoke.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_gpu_benchmark.ps1
```

Neu chay tren may khong co CUDA, GPU tests skip sach va benchmark ghi `skipped: no cuda`.

## 8. Ket qua benchmark

Artifacts chinh:

- `artifacts/cpu_baseline_metrics.json`
- `artifacts/profile_summary.json`
- `artifacts/environment_check.json`
- `artifacts/benchmarks.csv`
- `artifacts/benchmark_summary.json`
- `artifacts/gpu_benchmark_summary.json`

CPU baseline synthetic small: 1000 nodes, 5000 edges, 16 iterations, elapsed `0.08893630000056874s`, rank sum `1.0`.

GPU benchmark da chay that tren RTX 3060 Laptop GPU cho `v1`, `v2`, `v3_pull`, `v3_push`. Synthetic small graph qua nho nen GPU chua nhanh hon CPU do launch/JIT overhead, nhung relative error so voi CPU o muc ~1e-16.

## 9. Huong nop bai

Chay `pytest -q`, baseline/profile/benchmark, kiem tra `docs/submission_checklist.md`, roi nop repo GitHub cung proposal trong `docs/project_proposal.md`.
