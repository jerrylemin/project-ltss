"""Run the offline PageRank dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LTSS PageRank dashboard.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind on 127.0.0.1.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.ui.dashboard_server import create_app

    print(f"Dashboard running at http://127.0.0.1:{args.port}")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
