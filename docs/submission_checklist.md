# Submission Checklist

Last updated: 2026-05-29T18:31:00+07:00

- [x] README includes C3 PageRank title, Graph and Sparse track, team, setup, CUDA verification, CPU/test/benchmark/dashboard commands, benchmark results, correctness statement, push-vs-pull analysis, and final checklist.
- [x] `requirements.txt` uses exact `==` pins and supports a fresh dashboard setup.
- [x] `src/cpu_baseline.py --graph data/graphs/roadNet-CA.tsv` runs exit 0.
- [x] `python -m pytest tests/ -v` passes: 21 passed.
- [x] `python src/benchmark.py` writes `artifacts/benchmark_results.csv` with 25 data rows.
- [x] Benchmark graphs include roadNet-CA, com-youtube, wiki-talk, amazon0601, and soc-livejournal.
- [x] Benchmark versions include cpu_numpy, gpu_v1, gpu_v2, gpu_v3_pull, and gpu_v3_push.
- [x] `com-youtube` target met: `gpu_v3_push` at `0.127168s` <= 5 seconds.
- [x] Correctness vs SciPy is below `1e-6` for successful rows.
- [x] GPU V3 source contains `cuda.shfl_down_sync`.
- [x] Dashboard starts with `python scripts/run_dashboard.py --port 8000`.
- [x] Dashboard APIs `/api/benchmark`, `/api/graphs`, `/api/status`, and `/api/shfl_check` return HTTP 200.
- [x] `docs/final_pdf_compliance_audit.md` and `docs/benchmark_report.md` match current artifacts.
- [x] Downloaded SNAP graph TSV files are not staged.
