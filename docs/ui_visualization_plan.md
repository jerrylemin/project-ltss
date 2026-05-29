# UI Visualization Plan

## 1. Purpose and scope

This dashboard is an offline visualization surface for the LTSS C3 PageRank CUDA project. It presents the local benchmark CSV, graph-file presence, architecture notes, algorithm comparison, push-vs-pull explanation, and a demo-oriented verification checklist. It is intended for live grading and presentation sessions where clarity matters more than changing the benchmark workflow.

## 2. Tech stack decision

The dashboard uses FastAPI `0.136.3` and uvicorn `0.40.0`, with plain static HTML/CSS/vanilla JavaScript. These runtime dependencies are now exact-pinned in `requirements.txt` along with their transitive packages, so a fresh `.venv` can run `scripts/run_dashboard.py`. No CDN is used, and the server does not require CUDA or GPU availability.

## 3. File tree

```text
scripts/
  run_dashboard.py

src/ui/
  dashboard_server.py
  templates/
    index.html
  static/
    styles.css
    app.js

tests/
  test_dashboard.py
```

Documentation touched by this feature:

```text
docs/ui_visualization_plan.md
docs/session_handoff.md
README.md
docs/feature_progress.md
```

## 4. API contract

`GET /` returns the static dashboard HTML.

`GET /api/benchmark` returns JSON from `load_benchmark_csv("artifacts/benchmark_results.csv")`. The response is `{ "status": "ok", "rows": [...] }`, `{ "status": "missing", "rows": [] }`, or `{ "status": "error", "rows": [], "error": "..." }`. Missing files are expected state and still return HTTP 200.

`GET /api/graphs` returns `{ "status": "ok", "files": [...] }` when `data/graphs/` exists, or `{ "status": "missing", "files": [] }` when it does not.

`GET /api/status` returns structured local inspection metadata for the overview and checklist: benchmark CSV presence, row count, columns, mtime, graph files, source checks, and checklist booleans.

`GET /api/shfl_check` searches local CUDA/CUH/Python source files under `src/` for `shfl_down_sync` and returns `{ "status": "ok", "found": true|false, "matches": [...] }`.

## 5. Section inventory

Overview shows the project topic, track, target graph, correctness target, benchmark CSV presence, graph-file list, and benchmark CSV modification time.

Architecture renders the data pipeline as local HTML boxes, from SNAP TSV input through CSR construction, CPU/SciPy paths, GPU V1/V2/V3 paths, benchmark orchestration, and CSV output.

Data Structures shows incoming and outgoing CSR arrays, plus the per-node `out_degree[v]` vector and ping-pong rank buffers.

Algorithm Comparison summarizes CPU CSR PageRank, GPU V1, GPU V2, GPU V3 pull, and GPU V3 push with memory access, synchronization, and bottleneck notes.

Push vs Pull compares destination-owned gather against source-owned scatter and explains why atomics can dominate on hub-heavy graphs.

Benchmark Results fetches `/api/benchmark`, normalizes the current per-version CSV schema into a presentation table, and marks tolerance and com-youtube target status from real CSV fields or directly derived error values.

Verification Checklist combines auto-detected CSV evidence with manual command notes for pytest and V1 host-copy review.

Presentation Mode hides the dashboard shell and shows five short panels for problem, approach, correctness, performance, and current status.

## 6. Smoke test result

`python scripts/run_dashboard.py --port 8000` was started locally. `/`, `/api/benchmark`, `/api/graphs`, `/api/status`, and `/api/shfl_check` all returned HTTP 200. `/api/shfl_check` reported `found=true` with `src/gpu/pagerank_v3.py`.

## 7. What the dashboard does NOT do

The dashboard does not run benchmarks, start CUDA work, modify CUDA code, create fake benchmark values, or write to `artifacts/benchmark_results.csv`. It only reads existing local files and displays their current state.

## 8. Known limitations

The dashboard does not persist pytest results, so the pytest checklist item remains a manual command. The V1 host-copy loop inspection also remains manual because reliable loop-level static analysis is out of scope. The current workspace has `artifacts/benchmark_results.csv` present with 25 rows and `data/graphs/` present with all five required TSV files.
