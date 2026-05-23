from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def _latest_cuda_toolkit_path() -> str | None:
    candidates: list[Path] = []
    env_path = os.environ.get("CUDA_PATH")
    if env_path:
        candidates.append(Path(env_path))

    root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if root.exists():
        candidates.extend(path for path in root.iterdir() if path.is_dir() and path.name.startswith("v"))

    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return str(sorted(existing, key=lambda path: path.name)[-1])


def ensure_cuda_path() -> str | None:
    cuda_path = _latest_cuda_toolkit_path()
    if not cuda_path:
        return None

    os.environ.setdefault("CUDA_PATH", cuda_path)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    prepend = [
        str(Path(cuda_path) / "bin"),
        str(Path(cuda_path) / "bin" / "x64"),
    ]
    for item in reversed(prepend):
        if item not in path_parts and Path(item).exists():
            path_parts.insert(0, item)
    os.environ["PATH"] = os.pathsep.join(path_parts)
    return cuda_path


def _command_found(command: str) -> bool:
    return shutil.which(command) is not None


def _command_output(command: list[str], timeout_seconds: int = 30) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
    except Exception:
        return None
    output = (completed.stdout + completed.stderr).strip()
    return output or None


def _package_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def cuda_available() -> bool:
    ensure_cuda_path()
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def require_cuda() -> None:
    if not cuda_available():
        raise RuntimeError("CUDA is not available; GPU PageRank should be skipped.")


def cuda_status() -> dict[str, object]:
    cuda_path = ensure_cuda_path()
    status: dict[str, object] = {
        "nvidia_smi_found": _command_found("nvidia-smi"),
        "nvcc_found": _command_found("nvcc"),
        "cuda_path": cuda_path,
        "numba_version": _package_version("numba"),
        "numba_cuda_package_found": _package_version("numba-cuda") is not None,
        "numba_cuda_version": _package_version("numba-cuda"),
        "cuda_available": False,
        "device_name": None,
        "error": None,
    }
    if status["nvidia_smi_found"]:
        status["nvidia_smi"] = _command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        if status.get("nvidia_smi"):
            status["device_name"] = str(status["nvidia_smi"]).split(",", 1)[0].strip()
    if status["nvcc_found"]:
        status["nvcc_version"] = _command_output(["nvcc", "-V"])
    try:
        from numba import cuda

        available = bool(cuda.is_available())
        status["cuda_available"] = available
    except Exception as exc:
        status["error"] = str(exc)
    return status
