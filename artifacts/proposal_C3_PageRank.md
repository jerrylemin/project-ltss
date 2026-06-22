# GPU-Accelerated PageRank on Large Sparse Graphs

## Course and Group

Course: CSC14116 Applied Parallel Programming.  
Track: C3 PageRank, graph and sparse workloads.  
Group name: GPU PageRank Team.  
Repository: https://github.com/jerrylemin/project-ltss.

| Student ID | Member | Role |
|---|---|---|
| 21127645 | Le Minh | Team lead; CPU baseline, integration, documentation, GitHub |
| 21127224 | Nguyen Vu Bach | GPU kernels, benchmark review, testing, presentation support |


Keywords: PageRank; Sparse Matrix-Vector Multiplication; CUDA; Numba; Graph Analytics.

## Problem Statement

PageRank estimates the importance of nodes in a directed graph by repeatedly propagating rank mass through outgoing links until the rank vector stabilizes. For large web, road, product, and social graphs, each iteration touches millions of sparse edges, so total runtime is dominated by repeated sparse matrix-vector multiplication. The workload also includes irregular memory access through CSR rows, dangling-node rank redistribution, and a global L1 convergence reduction after every iteration. These properties make PageRank a strong GPU target: edge and row operations expose abundant data parallelism, while reductions and memory movement define the main optimization challenge.

## Dataset and Input

SNAP source: https://snap.stanford.edu/data/. The repository downloader is `scripts/download_graphs.py`; normalized TSV files are stored under `data/graphs/` and ignored by git. Loading parses edge lists, preserves directed edges, self-loops, and duplicate edges, then builds incoming CSR, outgoing CSR, and out-degree arrays.

| Graph | Nodes | Edges | Local input |
|---|---|---|---|
| roadNet-CA | 1,971,281 | 5,533,214 | data/graphs/roadNet-CA.tsv |
| com-Youtube | 1,157,828 | 2,987,624 | data/graphs/com-youtube.tsv |
| wiki-Talk | 2,394,385 | 5,021,410 | data/graphs/wiki-talk.tsv |
| amazon0601 | 403,394 | 3,387,388 | data/graphs/amazon0601.tsv |
| soc-LiveJournal1 | 4,847,571 | 68,993,773 | data/graphs/soc-livejournal.tsv |

## Why GPU-Suitable

- Each PageRank update for a node gathers contributions from incoming neighbors and can be mapped to independent GPU threads.
- CSR SpMV is naturally parallel over rows or edges, making it suitable for CUDA kernels when the graph is large enough.
- Pull mode gathers from incoming CSR and avoids write conflicts; push mode scatters along outgoing CSR and can improve locality on some graphs.
- The L1 convergence check can be reduced with block-level and warp-level reductions, including cuda.shfl_down_sync in V3.
- GPU acceleration targets the repeated edge traversal while CPU code remains useful for loading, SciPy reference checks, and reporting.

## Background

The project uses the PageRank recurrence r_new = alpha * A * r + alpha * dangling_mass / N + (1 - alpha) / N, followed by normalization. Here alpha is the damping factor, A is the link-transition matrix represented through CSR structures, and N is the number of nodes.

Dangling nodes have zero out-degree and cannot pass rank through outgoing edges, so their rank mass is redistributed uniformly at each iteration. Convergence is measured by the L1 norm between consecutive rank vectors, with tolerance 1e-6 in the project configuration and benchmark artifacts.

CSR stores sparse graph neighborhoods compactly using an indptr array and an indices array. The repository stores incoming CSR for pull/gather PageRank, outgoing CSR for push/scatter PageRank, and an out_degree array for normalizing each source contribution.

Pseudocode:

```text
Initialize rank vector uniformly.
Repeat until convergence or max_iter:
  compute dangling mass from zero-out-degree nodes.
  compute sparse matrix-vector PageRank contribution.
  apply damping and teleportation.
  compute L1 delta and normalize rank sum.
Return final rank vector.
```

## The Challenge

- Real graphs have irregular degree distributions, creating load imbalance between short rows and high-degree rows.
- Sparse CSR traversal causes non-contiguous rank-vector reads and limits memory locality.
- Push/scatter kernels require atomic updates and can suffer contention when many sources target the same destination.
- Every iteration needs a global L1 convergence reduction, so reduction overhead matters for small iteration times.
- Host-device transfer timing must be separated from kernel/convergence timing to make performance claims reproducible.
- Correctness must cover dangling nodes, self-loops, duplicate edges, disconnected components, and rank normalization.

## Resources

Hardware:

- Repository benchmark evidence records final validation on an NVIDIA GeForce RTX 3060 Laptop GPU, driver 591.86, 6144 MiB.
- The workspace uses Python 3.11.9 in .venv for the latest documented final run.
- Older environment_check.json evidence also records an RTX 3060 desktop-class run; the proposal treats the README/benchmark report as the latest repository summary.

Software:

- Python
- NumPy
- SciPy
- Numba CUDA
- pytest
- Jupyter Notebook
- pandas
- matplotlib
- Git and GitHub

Starter code and current modules:

- `src/data_loader.py`
- `src/pagerank_cpu.py`
- `src/cpu_baseline.py`
- `src/gpu/pagerank_v1.py`
- `src/gpu/pagerank_v2.py`
- `src/gpu/pagerank_v3.py`
- `src/benchmark.py`
- `tests/test_correctness.py`
- `scripts/download_graphs.py`
- `scripts/generate_final_report.py`

## Goals and Deliverables

| Level | Deliverables |
|---|---|
| 75 percent goal | CPU PageRank baseline; SciPy reference; CSR graph loader; basic correctness tests; small graph validation; one GPU kernel version. |
| 100 percent goal | GPU V1, V2, and V3; correctness within 1e-6 against SciPy; benchmark on five SNAP graphs; com-Youtube under 5 seconds; speedup table; runnable report notebook; DOCX proposal; README and reproducible commands. |
| 125 percent goal | Repeated benchmark statistics; separated transfer timing; offline dashboard or HTML visualization; extra graph analysis for push versus pull; self-loop regression tests; profiling charts. |

Demo deliverables:

- Benchmark table for five SNAP graphs.
- Speedup chart comparing CPU SpMV with GPU versions.
- Correctness table reporting relative L1 error against SciPy.
- Push-versus-pull comparison for V3.
- Final notebook and reproducible command list.

## Benchmark Evidence

Best final repeat-median GPU results from `artifacts/benchmark_results.csv`:

| Graph | Best GPU | Median GPU time (s) | Speedup vs CPU SpMV | Relative L1 vs SciPy |
|---|---|---|---|---|
| roadNet-CA | gpu_v3_push | 0.062436 | 85.521x | 1.533e-16 |
| com-Youtube | gpu_v3_push | 0.052838 | 12.255x | 3.291e-13 |
| wiki-Talk | gpu_v3_pull | 0.320790 | 11.399x | 2.601e-13 |
| amazon0601 | gpu_v3_push | 0.034461 | 71.090x | 3.214e-16 |
| soc-LiveJournal1 | gpu_v3_push | 1.141057 | 55.056x | 7.910e-15 |

Target: com-Youtube PageRank convergence at tolerance 1e-6 in <= 5 seconds. Repository evidence reports gpu_v3_push median 0.052838s, so the target is met in the measured workspace.

## 5-Week Weekly Schedule

Responsibilities are written as planned/reported contributions because git history does not independently prove balanced contribution by member.

| Week | Le Minh | Nguyen Vu Bach |
|---|---|---|
| Week 01, from [fill date] to [fill date] | Topic analysis, PageRank formula, C3 requirement mapping, initial CPU baseline review. | Dataset research, SNAP graph list, loader plan, repository structure and test plan. |
| Week 02, from [fill date] to [fill date] | CPU CSR PageRank implementation, SciPy reference comparison, convergence rule. | Graph preprocessing, CSR incoming/outgoing arrays, dangling-node and self-loop test cases. |
| Week 03, from [fill date] to [fill date] | GPU V1 CSR SpMV and damping update, CUDA kernel validation, pytest integration. | GPU V2 fused kernel, L1 convergence reduction, profiling and timing instrumentation. |
| Week 04, from [fill date] to [fill date] | GPU V3 pull implementation, warp-level reduction analysis, correctness comparison. | GPU V3 push implementation, atomic scatter path, push-versus-pull benchmark design. |
| Week 05, from [fill date] to [fill date] | Final report sections, CPU/GPU design explanation, correctness table, references. | Benchmark artifacts, speedup charts, README commands, DOCX proposal formatting. |

## References

1. Page, L., Brin, S., Motwani, R., and Winograd, T. The PageRank Citation Ranking: Bringing Order to the Web. Stanford InfoLab, 1999.
2. McLaughlin, A., and Bader, D. A. Scalable and High Performance PageRank on the GPU. SC14, 2014.
3. Stanford Network Analysis Project (SNAP) graph dataset collection. https://snap.stanford.edu/data/
4. SciPy sparse matrix documentation. https://docs.scipy.org/doc/scipy/reference/sparse.html
5. Numba CUDA documentation. https://numba.readthedocs.io/en/stable/cuda/
6. NVIDIA CUDA C Best Practices Guide.
7. GraphBLAS C API specification.

## Evidence Note

The requested template file `CSC14116 - Proposal.docx` was not present in the repository or attachment directory, so the DOCX was created from scratch using the same requested proposal sections.

Files supporting the proposal:

| Repo file | How it was used |
|---|---|
| README.md | Project description, setup commands, PageRank semantics, benchmark methodology, final results, hardware. |
| project_spec.yaml | C3 track, algorithm parameters, target graph, target time, required SNAP graph list. |
| src/data_loader.py | Edge-list parsing, incoming CSR, outgoing CSR, out-degree arrays, self-loop and duplicate-edge preservation. |
| src/pagerank_cpu.py | Custom NumPy CSR CPU PageRank and SciPy sparse reference implementation. |
| src/gpu/pagerank_v1.py | GPU V1 CSR SpMV and separate damping update kernels. |
| src/gpu/pagerank_v2.py | GPU V2 fused SpMV, damping, and block-level L1 convergence reduction. |
| src/gpu/pagerank_v3.py | GPU V3 pull/push kernels, outgoing CSR scatter, atomic adds, warp shuffle reductions. |
| src/benchmark.py | Five-graph benchmark harness, repeat/warmup methodology, SciPy comparison, transfer timing fields. |
| tests/test_correctness.py | Correctness tests against SciPy, dangling node tests, self-loop CPU/GPU regressions, push/pull equivalence. |
| docs/benchmark_report.md | Dataset sizes, final benchmark table, push-versus-pull discussion, limitations. |
| docs/team_plan.md | Reported responsibility split and explicit caveat that git history does not prove balanced contribution. |
| artifacts/benchmark_results.csv | Measured rows for CPU and GPU versions on five SNAP graphs, including relative L1 error. |
| artifacts/profile_summary.json | Real-graph CPU profile showing SpMV as the dominant measured cost. |
| notebooks/final_report.ipynb | Executable report notebook listed as a final deliverable. |

Contribution wording is conservative: the proposal uses reported/planned responsibility language for member work because `docs/team_plan.md` states that git history does not independently prove balanced contribution by member.

Dates needing confirmation: the official Week 01 through Week 05 date ranges remain `[fill date]` placeholders because the repository has commit dates but no course schedule dates.

Name normalization note: the attached prompt contained mojibake spellings `L?? Minh` and `Nguy???n V?? B??ch`; this proposal uses the ASCII forms already present in the repository (`Le Minh`, `Nguyen Vu Bach`) to avoid submitting corrupted text. If official accented spellings are required, replace them with `L? Minh` and `Nguy?n V? B?ch`.

Hardware note: README and docs/benchmark_report.md are treated as the latest repository summary: RTX 3060 Laptop GPU, driver 591.86, 6144 MiB. `artifacts/environment_check.json` contains older hardware evidence and is not used as the primary final hardware claim.
