"""Tests for build_sparse_optimizer contract."""

from __future__ import annotations

import pytest
import torch

from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.runtime.execution import build_sparse_optimizer


def test_default_path_builds_custom_sparse_adagrad() -> None:
    parameters = [torch.nn.Parameter(torch.randn(16, 4))]
    optimizer = build_sparse_optimizer(
        parameters,
        sparse_lr=0.05,
        sparse_weight_decay=0.0,
        runtime_execution=RuntimeExecutionConfig(),
        device="cpu",
    )
    assert type(optimizer).__name__ == "PCVRSparseAdagrad"


def test_nonzero_weight_decay_fails_early() -> None:
    parameters = [torch.nn.Parameter(torch.randn(16, 4))]
    with pytest.raises(ValueError, match="sparse_weight_decay"):
        build_sparse_optimizer(
            parameters,
            sparse_lr=0.05,
            sparse_weight_decay=0.1,
            runtime_execution=RuntimeExecutionConfig(),
            device="cpu",
        )
