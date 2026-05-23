from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def relative_l1_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.abs(expected).sum()) or 1.0
    return float(np.abs(actual - expected).sum() / denominator)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
