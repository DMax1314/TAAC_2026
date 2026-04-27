from __future__ import annotations

import ctypes
from functools import lru_cache
import importlib.util
import os
from typing import TYPE_CHECKING

import pytest
import torch


if TYPE_CHECKING:
    from tests.support import TestWorkspace


@lru_cache(maxsize=1)
def _has_cuda_driver_runtime() -> bool:
    if os.getenv("TAAC_FORCE_NO_LIBCUDA", "") == "1":
        return False
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    return True


@lru_cache(maxsize=1)
def _missing_cuda126_profile_packages() -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in ("torchrec", "fbgemm_gpu"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return tuple(missing)


@pytest.fixture
def benchmark_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def require_torchrec_runtime() -> None:
    missing_packages = _missing_cuda126_profile_packages()
    if missing_packages:
        missing_rendered = ", ".join(missing_packages)
        pytest.skip(
            "TorchRec-backed CPU benchmarks require the CUDA profile cuda126; missing packages: "
            f"{missing_rendered}"
        )
    if not _has_cuda_driver_runtime():
        pytest.skip(
            "TorchRec-backed CPU benchmarks require libcuda.so.1 when a CUDA profile is installed"
        )


@pytest.fixture
def benchmark_workspace(tmp_path_factory: pytest.TempPathFactory) -> TestWorkspace:
    # Keep TorchRec-backed workspace support out of the shared benchmark import path.
    from tests.support import create_test_workspace

    return create_test_workspace(tmp_path_factory.mktemp("bench_workspace"))