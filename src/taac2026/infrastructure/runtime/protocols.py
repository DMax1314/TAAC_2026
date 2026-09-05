"""Runtime protocols for trainer/model collaboration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class SparseParameterModel(Protocol):
    def get_sparse_params(self) -> Sequence[nn.Parameter]:
        ...

    def get_dense_params(self) -> Sequence[nn.Parameter]:
        ...


@runtime_checkable
class ReinitializableSparseParameterModel(SparseParameterModel, Protocol):
    def reinit_high_cardinality_params(self, cardinality_threshold: int = 10000) -> set[int]:
        """Reinitialize parameters in place and return their data pointers."""


__all__ = ["ReinitializableSparseParameterModel", "SparseParameterModel"]
