# Graph Report - project-ltss  (2026-05-23)

## Corpus Check
- 52 files · ~9,171 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 260 nodes · 283 edges · 40 communities (30 shown, 10 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 44 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e0081f13`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]

## God Nodes (most connected - your core abstractions)
1. `LTSS Project 1 - C3 PageRank` - 19 edges
2. `cuda_status` - 12 edges
3. `cuda_status` - 12 edges
4. `cuda_status` - 12 edges
5. `main()` - 12 edges
6. `run_pagerank_cpu()` - 11 edges
7. `LTSS Project Proposal - C3 PageRank` - 11 edges
8. `make_synthetic_graph()` - 9 edges
9. `load_edge_list()` - 8 edges
10. `main()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `test_project_spec_required_fields_exist()` --calls--> `load_config()`  [INFERRED]
  tests/test_config.py → src/config.py
- `test_read_edge_list_with_comments()` --calls--> `load_edge_list()`  [INFERRED]
  tests/test_data_loader.py → src/data_loader.py
- `test_auto_remap_non_contiguous_node_ids()` --calls--> `load_edge_list()`  [INFERRED]
  tests/test_data_loader.py → src/data_loader.py
- `test_fixture_snap_sample_loads()` --calls--> `load_edge_list()`  [INFERRED]
  tests/test_data_loader.py → src/data_loader.py
- `test_rank_sum_and_non_negative_values()` --calls--> `make_synthetic_graph()`  [INFERRED]
  tests/test_correctness.py → src/data_loader.py

## Communities (40 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (15): Returns (rank, metrics) with the same shape as CPU output., run_pagerank_gpu(), _benchmark_graph(), _iterations_per_second(), _load_graphs(), main(), _parse_csv_arg(), relative_l1_error() (+7 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (14): alpha, converged, elapsed_seconds, graph_name, iterations, l1_delta, l1_error_vs_scipy, max_iter (+6 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (25): 1. Ten project va Topic C3 PageRank, 1. Tên project + Topic C3 PageRank, 2. Muc tieu va Partial GPU Principle, 2. Mục tiêu và Partial GPU Principle, 3. Cai dat, 3. Cài đặt, 4. Chay CPU baseline / profile / benchmark / tests, 4. Chạy CPU baseline / profile / benchmark / tests (+17 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (15): _command_found(), _command_output(), cuda_available(), cuda_status(), ensure_cuda_path(), _latest_cuda_toolkit_path(), _package_version(), require_cuda() (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.19
Nodes (9): algorithm_params(), load_config(), validate_config(), main(), write_json(), main(), _top_functions(), _write_bottleneck_doc() (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (11): converged, elapsed_seconds, graph_name, iterations, l1_delta, mode, num_edges, num_nodes (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (11): Bottleneck Analysis, Dataset Details, Division of Work, Final GPU Evidence, LTSS Project Proposal - C3 PageRank, Measured CPU Baseline Timing, Optimization Plan, Performance Target (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.35
Nodes (9): build_graph(), _graph_from_edges(), load_edge_list(), load_graph_from_config(), make_synthetic_graph(), test_auto_remap_non_contiguous_node_ids(), test_fixture_snap_sample_loads(), test_read_edge_list_with_comments() (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (17): cuda_status, cuda_available, cuda_path, device_name, error, numba_cuda_package_found, numba_cuda_version, numba_version (+9 more)

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (5): Handoff, Project Codex Working Agreements, Repository Discipline, Runtime Expectations, Scope

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (5): Benchmark Report, code:powershell (.\.venv\Scripts\python.exe src/benchmark.py --config project), CUDA Status, Current Results, SNAP Status

### Community 11 - "Community 11"
Cohesion: 0.50
Nodes (3): Bottleneck Decision, Profile Summary, Top Functions by Cumulative Time

### Community 12 - "Community 12"
Cohesion: 0.50
Nodes (3): Completed, Feature Progress, Pending

### Community 14 - "Community 14"
Cohesion: 0.50
Nodes (3): code:powershell (python -m venv .venv), code:powershell (powershell -ExecutionPolicy Bypass -File scripts/download_sn), Setup and Run

### Community 31 - "Community 31"
Cohesion: 0.12
Nodes (16): checks, cuda_path, cuda_status, cuda_available, cuda_path, device_name, error, numba_cuda_package_found (+8 more)

### Community 32 - "Community 32"
Cohesion: 0.14
Nodes (13): cuda_status, cuda_available, cuda_path, device_name, error, numba_cuda_package_found, numba_cuda_version, numba_version (+5 more)

### Community 33 - "Community 33"
Cohesion: 0.22
Nodes (8): code:powershell (powershell -ExecutionPolicy Bypass -File scripts/download_sn), code:powershell (powershell -ExecutionPolicy Bypass -File scripts/download_sn), code:powershell (powershell -ExecutionPolicy Bypass -File scripts/download_sn), code:powershell (powershell -ExecutionPolicy Bypass -File scripts/download_sn), Helper Script, SNAP Dataset Setup, SNAP Sources, Supported Local Filenames

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (7): code:powershell (winget search Nvidia.CUDA), code:powershell ($env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Tool), Commands Used, CUDA Setup, Current Status, Terminal Reload Note, Troubleshooting

## Knowledge Gaps
- **122 isolated node(s):** `cuda_available`, `cuda_path`, `device_name`, `error`, `numba_cuda_package_found` (+117 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 0` to `Community 4`, `Community 7`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Why does `run_pagerank_cpu()` connect `Community 0` to `Community 4`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `main()` (e.g. with `load_config()` and `algorithm_params()`) actually correct?**
  _`main()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cuda_available`, `cuda_path`, `device_name` to the rest of the system?**
  _125 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.08923076923076922 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._