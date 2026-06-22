# Proposal Generation Notes

Last updated: 2026-06-23T02:03:16+07:00

## Output

- `artifacts/proposal_C3_PageRank.docx`
- `artifacts/proposal_C3_PageRank.md`

## Template Status

The requested template file `CSC14116 - Proposal.docx` was not found in the repository or attachment directory. The DOCX was therefore created from scratch with the requested structure: project information, members, keywords, references, problem statement, dataset/input, GPU suitability, background, challenge, resources, goals and deliverables, benchmark evidence, and a 5-week schedule.

## Evidence Used

- `README.md`: project description, hardware, setup, PageRank semantics, benchmark methodology, final result summary.
- `project_spec.yaml`: C3 PageRank track, algorithm parameters, target graph, target seconds, SNAP graph names.
- `docs/benchmark_report.md`: dataset sizes, final repeat benchmark table, push-versus-pull explanation, limitations.
- `docs/team_plan.md`: reported responsibility split and warning that git history does not independently prove balanced contribution.
- `src/data_loader.py`: edge-list parsing and incoming/outgoing CSR construction.
- `src/pagerank_cpu.py` and `src/cpu_baseline.py`: CPU NumPy CSR baseline, SciPy reference, profiling output path.
- `src/gpu/pagerank_v1.py`, `src/gpu/pagerank_v2.py`, `src/gpu/pagerank_v3.py`: GPU V1/V2/V3 implementation details.
- `src/benchmark.py`: repeat/warmup benchmark harness, SciPy comparison, transfer timing fields.
- `tests/test_correctness.py` and `tests/test_data_loader.py`: SciPy correctness, dangling-node, self-loop, outgoing CSR tests.
- `artifacts/benchmark_results.csv`, `artifacts/benchmark_summary.json`, `artifacts/profile_summary.json`: measured benchmark/profile evidence.
- `notebooks/final_report.ipynb`: executable final report deliverable.
- `graphify-out/GRAPH_REPORT.md`: graph-first navigation summary before source scan.

## Conservative Wording

The proposal does not say that commit history proves balanced contribution. Weekly responsibilities are written as planned/reported contributions because `docs/team_plan.md` explicitly says visible git history does not independently prove Nguyen Vu Bach's contribution by separate author identity.

## Remaining Confirmations

- Official Week 01 through Week 05 calendar dates.
- Whether the group name should remain `GPU PageRank Team`.
- Whether official student-name spelling should use accents (`Le Minh`/`Nguyen Vu Bach` are the repository spellings; the prompt text had mojibake).
- Whether the repo URL in `docs/project_proposal.md` is the final submission URL.
- Whether the latest primary hardware claim should remain the README/benchmark-report RTX 3060 Laptop GPU entry.
