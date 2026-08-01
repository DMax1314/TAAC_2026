"""Shared TileLang runtime discovery and compatibility helpers."""

from __future__ import annotations

import torch

try:
    import tilelang as tl  # type: ignore[import-not-found]
    import tilelang.language as T  # type: ignore[import-not-found]
except ImportError:
    tl = None
    T = None


def tilelang_available() -> bool:
    return tl is not None and T is not None


def cuda_multiprocessor_count(device: torch.device | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None
    resolved_device = device
    if resolved_device is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    if resolved_device.type != "cuda":
        return None
    try:
        return int(torch.cuda.get_device_properties(resolved_device).multi_processor_count)
    except Exception:
        return None


def tilelang_dtype(dtype: torch.dtype):
    if T is None:
        raise RuntimeError("tilelang language module is unavailable")
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.bfloat16:
        return T.bfloat16
    if dtype == torch.float32:
        return T.float32
    raise RuntimeError(f"tilelang kernels do not support dtype {dtype}")


__all__ = [
    "T",
    "cuda_multiprocessor_count",
    "tilelang_available",
    "tilelang_dtype",
    "tl",
]
