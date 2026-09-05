"""Embedding Adagrad with per-element accumulators and indexed sparse updates.

Duplicate rows are coalesced before squaring. Already-coalesced gradients from
the trainer's norm clipping are reused without another merge. Sparse updates
touch only indexed rows; dense embedding gradients use the same Adagrad math.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class PCVRSparseAdagrad:
    """Adagrad for sparse COO gradients (embedding tables)."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        eps: float = 1e-10,
        initial_accumulator_value: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"initial_accumulator_value must be >= 0, got {initial_accumulator_value}")
        self.param_groups: list[dict[str, object]] = [{
            "params": list(params),
            "lr": float(lr),
            "eps": float(eps),
            "initial_accumulator_value": float(initial_accumulator_value),
        }]
        self.state: dict[nn.Parameter, dict[str, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.grad = None

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = float(group["lr"])
            eps = float(group["eps"])
            initial_accumulator_value = float(group["initial_accumulator_value"])
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse and grad._nnz() == 0:
                    continue
                if grad.is_sparse and (parameter.ndim != 2 or grad.sparse_dim() != 1):
                    raise ValueError("PCVRSparseAdagrad requires row-sparse COO embedding gradients")

                state = self.state.setdefault(parameter, {})
                if "sum" not in state:
                    state["sum"] = torch.full_like(parameter, initial_accumulator_value)
                accumulator = state["sum"]
                if grad.is_sparse:
                    grad = grad.coalesce()
                    rows, values = grad.indices()[0], grad.values()
                    accumulator.index_add_(0, rows, values.square())
                    std = accumulator.index_select(0, rows).sqrt_().add_(eps)
                    parameter.index_add_(0, rows, (values / std).mul_(-lr))
                else:
                    accumulator.addcmul_(grad, grad)
                    std = accumulator.sqrt().add_(eps)
                    parameter.addcdiv_(grad, std, value=-lr)


__all__ = ["PCVRSparseAdagrad"]
