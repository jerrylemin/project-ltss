"""Backend helpers and routes for the offline PageRank dashboard."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CSV = PROJECT_ROOT / "artifacts" / "benchmark_results.csv"
GRAPH_DIR = PROJECT_ROOT / "data" / "graphs"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_benchmark_csv(path: str) -> dict:
    """
    Returns:
      { "status": "ok", "rows": [...] }       if file exists and parses cleanly
      { "status": "missing", "rows": [] }     if file does not exist
      { "status": "error", "error": str }     if file exists but is malformed
    Never raises. Never returns fake data.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return {"status": "missing", "rows": []}

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header or any(not column.strip() for column in header):
                return {"status": "error", "rows": [], "error": "CSV header is missing or invalid"}

            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if not values or all(value == "" for value in values):
                    continue
                if len(values) != len(header):
                    message = (
                        f"CSV row {line_number} has {len(values)} fields; "
                        f"expected {len(header)}"
                    )
                    return {"status": "error", "rows": [], "error": message}
                rows.append(dict(zip(header, values)))
    except Exception as exc:
        return {"status": "error", "rows": [], "error": str(exc)}

    return {"status": "ok", "rows": rows}


def detect_graph_files(graph_dir: str) -> dict:
    """
    Returns:
      { "status": "ok", "files": ["roadNet-CA.tsv", ...] }
      { "status": "missing", "files": [] }    if dir does not exist
    """
    path = Path(graph_dir)
    if not path.exists():
        return {"status": "missing", "files": []}

    files = sorted(item.name for item in path.iterdir() if item.is_file() and item.suffix == ".tsv")
    return {"status": "ok", "files": files}


def check_shfl_down_sync(source_dir: str) -> dict:
    """Return whether shfl_down_sync is present in local CUDA or Numba source files."""
    root = Path(source_dir)
    if not root.exists():
        return {"status": "missing", "found": False, "matches": []}

    matches: list[str] = []
    for suffix in (".cu", ".cuh", ".py"):
        for path in root.rglob(f"*{suffix}"):
            relative = path.relative_to(PROJECT_ROOT)
            if relative.parts[:2] == ("src", "ui"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "shfl_down_sync" in text:
                matches.append(str(relative))

    return {"status": "ok", "found": bool(matches), "matches": sorted(matches)}


def _file_mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return timestamp.isoformat()


def _benchmark_columns(path: Path) -> list[str]:
    parsed = load_benchmark_csv(str(path))
    rows = parsed.get("rows", [])
    if parsed.get("status") != "ok" or not rows:
        return []
    first_row = rows[0]
    return list(first_row.keys())


def _has_value(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    return value is not None and str(value).strip() not in {"", "nan", "NaN"}


def get_project_status() -> dict:
    """
    Returns structured metadata for the overview panel.
    All values come from actual file inspection, not hardcoded strings.
    """
    benchmark = load_benchmark_csv(str(BENCHMARK_CSV))
    rows = benchmark.get("rows", [])
    graph_result = detect_graph_files(str(GRAPH_DIR))
    shfl_result = check_shfl_down_sync(str(PROJECT_ROOT / "src"))

    roadnet_cpu = any(
        "roadnet-ca" in str(row.get("graph_name", row.get("graph", ""))).lower()
        and str(row.get("version", "")).lower() in {"cpu_numpy", "cpu"}
        and (_has_value(row, "convergence_time_s") or _has_value(row, "cpu_time_s"))
        for row in rows
    )
    youtube_row = any(
        "youtube" in str(row.get("graph_name", row.get("graph", ""))).lower()
        for row in rows
    )

    status = {
        "project_root": str(PROJECT_ROOT),
        "benchmark_csv": {
            "path": str(BENCHMARK_CSV.relative_to(PROJECT_ROOT)),
            "status": benchmark.get("status"),
            "present": BENCHMARK_CSV.exists(),
            "row_count": len(rows) if benchmark.get("status") == "ok" else 0,
            "columns": _benchmark_columns(BENCHMARK_CSV),
            "last_modified": _file_mtime_iso(BENCHMARK_CSV),
            "error": benchmark.get("error"),
        },
        "graph_files": graph_result,
        "source_checks": {
            "shfl_down_sync": shfl_result,
        },
        "checklist": {
            "roadnet_cpu_baseline_detected": roadnet_cpu,
            "full_benchmark_completed": len(rows) >= 2 if benchmark.get("status") == "ok" else False,
            "youtube_row_detected": youtube_row,
        },
    }
    return status


def create_app() -> FastAPI:
    app = FastAPI(title="LTSS PageRank Dashboard")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(TEMPLATE_DIR / "index.html")

    @app.get("/api/benchmark")
    def api_benchmark() -> JSONResponse:
        return JSONResponse(load_benchmark_csv(str(BENCHMARK_CSV)))

    @app.get("/api/graphs")
    def api_graphs() -> JSONResponse:
        return JSONResponse(detect_graph_files(str(GRAPH_DIR)))

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        return JSONResponse(get_project_status())

    @app.get("/api/shfl_check")
    def api_shfl_check() -> JSONResponse:
        return JSONResponse(check_shfl_down_sync(str(PROJECT_ROOT / "src")))

    return app
