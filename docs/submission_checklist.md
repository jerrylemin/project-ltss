# Submission Checklist

Last updated: 2026-06-13T00:00:00+07:00

- [x] README includes C3 PageRank title, Graph and Sparse track, team, setup, CUDA verification, CPU/test/benchmark/dashboard commands, benchmark results, correctness statement, push-vs-pull analysis, and final checklist.
- [x] `requirements.txt` uses exact `==` pins and supports a fresh dashboard setup.
- [x] `src/cpu_baseline.py --graph amazon0601 --profile --output artifacts/profile_summary.json` runs exit 0.
- [x] `python -m pytest tests/ -v` passes with self-loop CPU/GPU regressions.
- [x] `python src/benchmark.py --graphs all --repeat 5 --warmup 2 --output artifacts/benchmark_results.csv --include-transfer-timing` writes `artifacts/benchmark_results.csv` with 25 data rows and repeat statistics.
- [x] Benchmark graphs include roadNet-CA, com-youtube, wiki-talk, amazon0601, and soc-livejournal.
- [x] Benchmark versions include cpu_numpy, gpu_v1, gpu_v2, gpu_v3_pull, and gpu_v3_push.
- [x] `com-youtube` target met: `gpu_v3_push` repeat median `0.052838s` <= 5 seconds.
- [x] Self-loop and duplicate-edge semantics are preserved consistently.
- [x] Correctness vs SciPy is below `1e-6` for successful rows.
- [x] GPU V3 source contains `cuda.shfl_down_sync`.
- [x] Final report notebook executes with `jupyter nbconvert`.
- [x] Dashboard starts with `python scripts/run_dashboard.py --port 8000`.
- [x] Dashboard APIs `/api/benchmark`, `/api/graphs`, `/api/status`, and `/api/shfl_check` return HTTP 200.
- [x] `docs/final_pdf_compliance_audit.md` and `docs/benchmark_report.md` match current artifacts.
- [x] Downloaded SNAP graph TSV files are not staged.
