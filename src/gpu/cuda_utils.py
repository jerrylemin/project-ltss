from __future__ import annotations


def cuda_available() -> bool:
    try:
        from numba import cuda

        return bool(cuda.is_available())
    except Exception:
        return False


def require_cuda() -> None:
    if not cuda_available():
        raise RuntimeError("CUDA is not available; GPU PageRank should be skipped.")


def cuda_status() -> dict[str, object]:
    try:
        from numba import cuda

        available = bool(cuda.is_available())
        name = None
        if available:
            name = cuda.get_current_device().name.decode("utf-8", errors="ignore")
        return {"cuda_available": available, "device_name": name}
    except Exception as exc:
        return {"cuda_available": False, "device_name": None, "error": str(exc)}
