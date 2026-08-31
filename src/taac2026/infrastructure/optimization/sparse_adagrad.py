"""Sparse-gradient Adagrad with index-based updates.

``torch.optim.Adagrad`` routes sparse COO gradients through several sparse
primitives (``coalesce``, ``sparse_mask``, ``_make_sparse``, invariant
checks) that dominate the optimizer cost for large embedding tables on CUDA
(measured ~4x slower than the dense path on A30). This class implements the
same math exclusively with dense-index ops:

    state_sum[row]  += sum_d(grad[row, d] ** 2)
    param[row, d]   -= lr * grad[row, d] / sqrt(state_sum[row] + eps)

``scatter_reduce_`` (small tables) and ``index_add_`` (large tables) merge
duplicate rows, and ``index_add_`` applies the per-row update exactly once per
unique row, so the gradient does not need to be coalesced first.

Interface-compatible with ``torch.optim.Optimizer`` for the pieces the PCVR
trainer uses: ``param_groups``, ``state``, ``step()``, and ``zero_grad()``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

# Merge-strategy threshold: full-table scatter+any cost grows with the number
# of table elements (vocab x dim), while the sorted-segment merge stays at nnz
# scale and wins on big or wide tables (measured 3.4x faster on a 1M-row table,
# A30). Keep the small path only for tables within both the vocab and element
# budgets so wide embeddings do not allocate a full (vocab x dim) dense tensor.
_SCATTER_MERGE_VOCAB_LIMIT = 100_000
_SCATTER_MERGE_FULL_TABLE_ELEMENT_LIMIT = 25_600_000


class PCVRSparseAdagrad:
    """Adagrad for sparse COO gradients (embedding tables)."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
        initial_accumulator_value: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"initial_accumulator_value must be >= 0, got {initial_accumulator_value}")
        if weight_decay != 0.0:
            raise ValueError("PCVRSparseAdagrad does not support weight_decay (matches torch Adagrad sparse behavior)")
        self.defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "eps": float(eps),
            "initial_accumulator_value": float(initial_accumulator_value),
        }
        self.param_groups: list[dict[str, object]] = [{"params": list(params), **self.defaults}]
        self.state: dict[nn.Parameter, dict[str, object]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        del set_to_none  # interface parity with torch.optim.Optimizer; always clears
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad = None  # type: ignore[assignment]

    def _sparse_step(
        self,
        parameter: nn.Parameter,
        grad: torch.Tensor,
        lr: float,
        eps: float,
        initial_accumulator_value: float,
    ) -> None:
        state = self.state.setdefault(parameter, {})
        accumulator = state.get("sum")
        if accumulator is None:
            accumulator = torch.full_like(parameter, initial_accumulator_value)
            state["sum"] = accumulator

        indices = grad._indices()[0]
        grad_values = grad._values().to(parameter.dtype)
        if (
            parameter.shape[0] <= _SCATTER_MERGE_VOCAB_LIMIT
            and parameter.numel() <= _SCATTER_MERGE_FULL_TABLE_ELEMENT_LIMIT
        ):
            merged = torch.zeros(parameter.shape, device=parameter.device, dtype=parameter.dtype)
            merged.scatter_reduce_(
                0,
                indices.unsqueeze(1).expand(-1, parameter.shape[1]),
                grad_values,
                reduce="sum",
                include_self=False,
            )
            nonzero_rows = merged.any(dim=1)
            unique_rows = nonzero_rows.nonzero(as_tuple=False).squeeze(1)
            if unique_rows.numel() == 0:
                return
            merged_values = merged[unique_rows]
        else:
            # Large tables: sort row indices and sum per segment, keeping every
            # op at nnz scale instead of scanning the full (vocab, dim) table.
            unique_rows, seg_ids = indices.unique(return_inverse=True)
            merged_values = torch.zeros(
                (unique_rows.numel(), parameter.shape[1]),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            merged_values.index_add_(0, seg_ids, grad_values)
        accumulator.index_add_(0, unique_rows, merged_values.pow(2))
        std = accumulator.index_select(0, unique_rows).sqrt_().add_(eps)
        parameter.index_add_(0, unique_rows, (merged_values / std).mul_(-lr))

    def _dense_step(
        self,
        parameter: nn.Parameter,
        grad: torch.Tensor,
        lr: float,
        eps: float,
        initial_accumulator_value: float,
    ) -> None:
        state = self.state.setdefault(parameter, {})
        accumulator = state.get("sum")
        if accumulator is None:
            accumulator = torch.full_like(
                parameter,
                initial_accumulator_value,
            )
            state["sum"] = accumulator
        accumulator.addcmul_(grad, grad, value=1.0)
        std = accumulator.sqrt().add_(eps)
        parameter.addcdiv_(grad, std, value=-lr)

    @torch.no_grad()
    def step(self, closure: object = None) -> object:
        loss = None
        if closure is not None:
            raise TypeError("PCVRSparseAdagrad does not support closures")
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            eps = float(group["eps"])
            initial_accumulator_value = float(group["initial_accumulator_value"])
            if weight_decay != 0.0:
                raise RuntimeError("weight_decay is not compatible with sparse gradients")
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    if grad._nnz() == 0:
                        continue
                    self._sparse_step(parameter, grad, lr, eps, initial_accumulator_value)
                else:
                    self._dense_step(parameter, grad, lr, eps, initial_accumulator_value)
        return loss

    def __repr__(self) -> str:
        return f"PCVRSparseAdagrad(lr={self.defaults['lr']})"


__all__ = ["PCVRSparseAdagrad"]
