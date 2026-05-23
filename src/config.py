from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("project_spec.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = [
        ("project_id",),
        ("title",),
        ("algorithm", "alpha"),
        ("algorithm", "tolerance"),
    ]
    missing: list[str] = []
    for path in required:
        cursor: Any = config
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                missing.append(".".join(path))
                break
            cursor = cursor[key]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")


def algorithm_params(config: dict[str, Any]) -> tuple[float, float, int]:
    algorithm = config.get("algorithm", {})
    return (
        float(algorithm.get("alpha", 0.85)),
        float(algorithm.get("tolerance", 1e-6)),
        int(algorithm.get("max_iter", 200)),
    )


def synthetic_size(config: dict[str, Any], name: str) -> tuple[int, int]:
    sizes = config.get("dataset", {}).get("synthetic_sizes", {})
    if name not in sizes:
        raise ValueError(f"Unknown synthetic graph size: {name}")
    spec = sizes[name]
    return int(spec["nodes"]), int(spec["edges"])
