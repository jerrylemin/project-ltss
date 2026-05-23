from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import main as benchmark_main
from src.cpu_baseline import main as baseline_main
from src.profile_cpu import main as profile_main


def main() -> int:
    baseline_main([])
    profile_main([])
    benchmark_main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
