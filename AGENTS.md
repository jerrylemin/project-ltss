# Project Codex Working Agreements

## Scope
- Codex-only workflow for LTSS Project 1, topic C3 PageRank.
- Work on branch `main`; do not create feature branches.
- Do not use Claude Code, Claude-only commands, hooks, or `~/.claude`.

## Repository Discipline
- Read this file, `README.md`, and durable docs under `docs/` before substantial work.
- Preserve user changes. Do not delete existing code without a backup.
- Do not hardcode personal machine paths in production code.
- Never commit credentials or tokens.
- Run `pytest -q` before every commit.

## Runtime Expectations
- CPU baseline must run on machines without a GPU.
- CUDA/GPU paths must auto-detect availability and skip cleanly when CUDA is missing.
- Default dataset mode is synthetic; SNAP paths are optional overrides.

## Handoff
- Update `docs/session_handoff.md` and `docs/feature_progress.md` after meaningful work.
- Record unresolved small decisions in `docs/open_questions.md` instead of blocking.
