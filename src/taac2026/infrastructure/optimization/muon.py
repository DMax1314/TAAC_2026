"""Muon optimizer with AdamW fallback for non-matrix parameters."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


_MUON_BATCHED_MATRIX_ATTRIBUTE = "_taac_muon_batched_matrix"
_MUON_ADAMW_ATTRIBUTE = "_taac_muon_adamw"


def mark_muon_batched_matrix(parameter: torch.nn.Parameter) -> torch.nn.Parameter:
    """Mark ``[batch, rows, cols]`` matrices for independent Muon updates."""
    if parameter.ndim != 3:
        raise ValueError("Muon batched-matrix parameters must be three-dimensional")
    setattr(parameter, _MUON_BATCHED_MATRIX_ATTRIBUTE, True)
    return parameter


def mark_muon_adamw(parameter: torch.nn.Parameter) -> torch.nn.Parameter:
    """Force a dense parameter to use Muon's AdamW branch."""
    setattr(parameter, _MUON_ADAMW_ATTRIBUTE, True)
    return parameter


def _orthogonalize_update(
    update: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    original_dtype = update.dtype
    matrix = update.float().reshape(update.shape[0], -1)
    rows, cols = matrix.shape
    transposed = rows > cols
    if transposed:
        matrix = matrix.t()

    norm = matrix.norm()
    if not torch.isfinite(norm) or norm <= eps:
        return torch.zeros_like(update)

    matrix = matrix / norm.clamp_min(eps)
    for _ in range(max(0, steps)):
        gram = matrix @ matrix.t()
        matrix = 1.5 * matrix - 0.5 * gram @ matrix

    if transposed:
        matrix = matrix.t()
    return matrix.reshape_as(update).to(original_dtype)


def _orthogonalize_updates_batched(
    updates: list[torch.Tensor],
    *,
    steps: int,
    eps: float = 1e-12,
) -> list[torch.Tensor]:
    """Orthogonalize a batch of same-shape 2D updates with batched matmuls.

    Mathematically identical to calling ``_orthogonalize_update`` per tensor,
    but groups same-shaped matrices into one ``bmm`` chain so the per-matrix
    Newton-Schulz iterations collapse from many small GEMM launches into a few
    batched ones. ``steps <= 0`` skips the Newton-Schulz iterations entirely
    (pure normalization, used for shape-uniformity checks).
    """
    if not updates:
        return []
    if steps <= 0:
        return [_orthogonalize_update(update, steps=steps) for update in updates]

    original_dtypes = [update.dtype for update in updates]
    shapes = [tuple(update.reshape(update.shape[0], -1).shape) for update in updates]
    # 2D (rows, cols); matrices with rows > cols are transposed before iteration.
    matrices = []
    transposed_flags = []
    for update, shape in zip(updates, shapes, strict=True):
        matrix = update.float().reshape(shape[0], -1)
        transposed = shape[0] > shape[1]
        transposed_flags.append(transposed)
        matrices.append(matrix.t() if transposed else matrix)

    norms = torch.stack([matrix.norm() for matrix in matrices])
    finite_mask = torch.isfinite(norms) & (norms > eps)
    batched = torch.stack(matrices)  # (N, R, C)
    batched = batched / norms.clamp_min(eps).unsqueeze(1).unsqueeze(2)
    for _ in range(max(1, steps)):
        gram = torch.bmm(batched, batched.transpose(1, 2))
        batched = 1.5 * batched - 0.5 * torch.bmm(gram, batched)

    results: list[torch.Tensor] = []
    for index, (update, _shape, transposed) in enumerate(
        zip(updates, shapes, transposed_flags, strict=True)
    ):
        if not finite_mask[index]:
            results.append(torch.zeros_like(update))
            continue
        matrix = batched[index]
        if transposed:
            matrix = matrix.t()
        results.append(matrix.reshape_as(update).to(original_dtypes[index]))
    return results


def _orthogonalize_batched_matrix_update(
    update: torch.Tensor,
    *,
    steps: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Orthogonalize each matrix in one ``[batch, rows, cols]`` tensor."""
    original_dtype = update.dtype
    matrices = update.float()
    transposed = matrices.shape[1] > matrices.shape[2]
    if transposed:
        matrices = matrices.transpose(1, 2)
    norms = torch.linalg.vector_norm(matrices, dim=(1, 2), keepdim=True)
    finite = torch.isfinite(norms) & (norms > eps)
    matrices = matrices / norms.clamp_min(eps)
    for _ in range(max(1, steps)):
        gram = torch.bmm(matrices, matrices.transpose(1, 2))
        matrices = 1.5 * matrices - 0.5 * torch.bmm(gram, matrices)
    matrices = torch.where(finite, matrices, torch.zeros_like(matrices))
    if transposed:
        matrices = matrices.transpose(1, 2)
    return matrices.to(original_dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        adamw_betas: tuple[float, float] = (0.9, 0.98),
        adamw_eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        beta1, beta2 = adamw_betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"adamw_betas must be in [0, 1), got {adamw_betas}")
        if adamw_eps <= 0.0:
            raise ValueError(f"adamw_eps must be > 0, got {adamw_eps}")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "weight_decay": float(weight_decay),
            "adamw_betas": (float(beta1), float(beta2)),
            "adamw_eps": float(adamw_eps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = float(group["adamw_eps"])

            matrix_params = [
                p
                for p in group["params"]
                if p.grad is not None
                and p.ndim >= 2
                and not bool(getattr(p, _MUON_ADAMW_ATTRIBUTE, False))
            ]
            vector_params = [
                p
                for p in group["params"]
                if p.grad is not None
                and (p.ndim < 2 or bool(getattr(p, _MUON_ADAMW_ATTRIBUTE, False)))
            ]

            prepared: list[tuple[torch.Tensor, torch.Tensor, float]] = []
            batched_prepared: list[tuple[torch.Tensor, torch.Tensor, float]] = []
            for parameter in matrix_params:
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[parameter]
                momentum_buffer = state.setdefault(
                    "momentum_buffer", torch.zeros_like(parameter)
                )
                momentum_buffer.mul_(momentum).add_(grad)
                update = (
                    grad.add(momentum_buffer, alpha=momentum)
                    if nesterov
                    else momentum_buffer
                )
                if bool(getattr(parameter, _MUON_BATCHED_MATRIX_ATTRIBUTE, False)):
                    if parameter.ndim != 3:
                        raise RuntimeError(
                            "Muon batched-matrix parameters must remain three-dimensional"
                        )
                    update_scale = math.sqrt(
                        max(1.0, parameter.shape[1] / max(1, parameter.shape[2]))
                    )
                    batched_prepared.append((parameter, update, update_scale))
                    continue
                matrix = update.reshape(update.shape[0], -1)
                update_scale = math.sqrt(
                    max(1.0, matrix.shape[0] / max(1, matrix.shape[1]))
                )
                prepared.append((parameter, update, update_scale))

            for parameter, update, update_scale in batched_prepared:
                orthogonal_update = _orthogonalize_batched_matrix_update(
                    update, steps=ns_steps
                )
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(orthogonal_update, alpha=-lr * update_scale)

            # Batch Newton-Schulz iterations by matrix shape: same-shaped updates
            # run as one bmm chain, single-shaped groups stay per-matrix.
            groups: dict[tuple[int, int], list[int]] = {}
            for index, (_parameter, update, _scale) in enumerate(prepared):
                shape = tuple(update.reshape(update.shape[0], -1).shape)
                groups.setdefault(shape, []).append(index)

            orthogonal_updates: list[torch.Tensor | None] = [None] * len(prepared)
            for shape, indices in groups.items():
                del shape
                group_updates = [prepared[index][1] for index in indices]
                if len(group_updates) == 1:
                    orthogonal_updates[indices[0]] = _orthogonalize_update(
                        group_updates[0], steps=ns_steps
                    )
                else:
                    results = _orthogonalize_updates_batched(
                        group_updates, steps=ns_steps
                    )
                    for index, result in zip(indices, results, strict=True):
                        orthogonal_updates[index] = result

            for (parameter, _update, update_scale), orthogonal_update in zip(
                prepared, orthogonal_updates, strict=True
            ):
                if orthogonal_update is None:
                    continue
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(orthogonal_update, alpha=-lr * update_scale)

            if not vector_params:
                continue

            # 1D/0D parameters follow AdamW; use foreach ops so the per-parameter
            # elementwise kernel cascade collapses into a handful of launches.
            grad_list = [p.grad for p in vector_params]
            state_list = [self.state[p] for p in vector_params]
            for p, state in zip(vector_params, state_list, strict=True):
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg_list = [state["exp_avg"] for state in state_list]
            exp_avg_sq_list = [state["exp_avg_sq"] for state in state_list]
            for state in state_list:
                state["step"] += 1
            step = int(state_list[0]["step"])
            if any(int(state["step"]) != step for state in state_list):
                raise RuntimeError("Muon vector parameters desynchronized across steps")

            if weight_decay != 0.0:
                torch._foreach_mul_(vector_params, 1.0 - lr * weight_decay)

            torch._foreach_mul_(exp_avg_list, beta1)
            torch._foreach_add_(exp_avg_list, grad_list, alpha=1.0 - beta1)
            torch._foreach_mul_(exp_avg_sq_list, beta2)
            torch._foreach_addcmul_(
                exp_avg_sq_list, grad_list, grad_list, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom_list = [exp_avg_sq.sqrt() for exp_avg_sq in exp_avg_sq_list]
            torch._foreach_div_(denom_list, math.sqrt(bias_correction2))
            torch._foreach_add_(denom_list, adamw_eps)
            torch._foreach_addcdiv_(
                vector_params, exp_avg_list, denom_list, value=-lr / bias_correction1
            )

        return loss


__all__ = ["Muon", "mark_muon_adamw", "mark_muon_batched_matrix"]
