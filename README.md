# LTSS Project 1 - C3 PageRank

## 1. Tên project + Topic C3 PageRank

Project LTSS triển khai PageRank cho topic C3, track Graph and Sparse. Repo mục tiêu: `https://github.com/jerrylemin/project-ltss`.

## 2. Mục tiêu và Partial GPU Principle

Mục tiêu là đo CPU baseline PageRank trên đồ thị sparse, xác định bottleneck SpMV trong vòng lặp PageRank, rồi tăng tốc phần kernel lặp bằng GPU. Các bước đọc dữ liệu, cấu hình, validate, báo cáo và fallback vẫn chạy trên CPU để đảm bảo chương trình dùng được trên máy không có GPU.

## 3. Cài đặt

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chạy CPU baseline / profile / benchmark / tests

```powershell
python src/cpu_baseline.py --config project_spec.yaml
python src/profile_cpu.py --config project_spec.yaml
python src/benchmark.py --config project_spec.yaml
pytest -q
```

## 5. Dataset: SNAP và synthetic fallback

Mặc định project dùng synthetic graph trong `project_spec.yaml`, nên không cần dataset thật để chạy. Nếu có SNAP edge list local, đặt đường dẫn vào `dataset.edges_path` hoặc chạy:

```powershell
python src/cpu_baseline.py --edges data/raw/com-Youtube.txt
```

## 6. Cấu trúc repo

```text
src/        code CPU, profile, benchmark, GPU optional
tests/      pytest suite
docs/       proposal, benchmark report, handoff, checklist
artifacts/  metrics JSON, profile summary, benchmark CSV
scripts/    helper PowerShell scripts
```

## 7. Trạng thái GPU (auto-detect)

GPU được detect bằng Numba CUDA. Nếu không có CUDA runtime tương thích, GPU benchmark ghi row `skipped: no cuda` và GPU tests skip sạch, không làm fail pipeline CPU.

## 8. Kết quả benchmark

Kết quả được sinh sau khi chạy:

- `artifacts/cpu_baseline_metrics.json`
- `artifacts/profile_summary.json`
- `artifacts/benchmarks.csv`
- `artifacts/benchmark_summary.json`

## 9. Hướng nộp bài

Chạy `pytest -q`, chạy baseline/profile/benchmark, kiểm tra `docs/submission_checklist.md`, rồi nộp repo GitHub cùng proposal trong `docs/project_proposal.md`.
