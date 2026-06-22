# Team Plan and Evidence

Last updated: 2026-06-13

## Visible Git Evidence

`git shortlog -sne --all` currently shows:

| Visible author | Commits | Notes |
|---|---:|---|
| `jerrylemin <jerryle.minh.3@gmail.com>` | 16 | Main historical commit identity. |
| `Le Minh <jerryle.minh.3@gmail.com>` | 3 | Same email family, different display name. |

This git history does not independently prove balanced contribution by member. The team should explain during presentation that some work was integrated through one machine/account and identify live ownership by module and presentation section.

## Contribution Table

| Member | Responsibility | Files or modules | Evidence |
|---|---|---|---|
| Le Minh | CPU baseline, integration, documentation, GitHub, final artifact generation | `src/pagerank_cpu.py`, `src/cpu_baseline.py`, `README.md`, `docs/`, `artifacts/` | Visible git author identities and final integration commits. |
| Nguyen Vu Bach | GPU kernels, benchmark review, correctness testing, performance discussion | `src/gpu/pagerank_v1.py`, `src/gpu/pagerank_v2.py`, `src/gpu/pagerank_v3.py`, `src/benchmark.py`, `tests/test_correctness.py` | Responsibility stated in project docs; git attribution is not visible by separate author identity. |

## Presentation Ownership

| Presentation section | Owner |
|---|---|
| CPU baseline and profiling | Le Minh |
| GPU V1 and V2 | Nguyen Vu Bach |
| GPU V3 push/pull | Nguyen Vu Bach |
| Benchmark methodology and report | Le Minh |
| Correctness and tests | Both |
| Dataset setup and reproducibility | Le Minh |

## What Is Unclear

- There is no separate commit author identity for Nguyen Vu Bach in this repository.
- Several commits use generic messages such as `update`, so commit titles alone do not describe ownership well.
- The team should be ready to explain pair-programming, local integration, or account-sharing workflow if graders ask about git balance.
