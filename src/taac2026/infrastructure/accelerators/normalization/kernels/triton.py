"""Triton RMSNorm kernel builders."""

from __future__ import annotations

import torch

from taac2026.infrastructure.accelerators.triton_runtime import (
    tl,
    triton,
    triton_available,
    triton_next_power_of_2,
    triton_num_warps,
)


def _ensure_triton() -> None:
    if not triton_available():
        raise RuntimeError("triton is not installed")


def build_rms_norm_forward_kernel(
    rows: int,
    cols: int,
    block_rows: int,
    eps: float,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def rms_norm_forward_kernel(
        x,
        weight,
        out,
        inv_rms,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)
        x_values = tl.load(x + row_offsets[:, None] * COLS + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)
        row_scale = tl.rsqrt(tl.sum(x_values * x_values, axis=1) / COLS + EPS)
        out_values = x_values * row_scale[:, None] * weight_values[None, :]
        tl.store(out + row_offsets[:, None] * COLS + col_offsets[None, :], out_values, mask=mask)
        tl.store(inv_rms + row_offsets, row_scale, mask=row_offsets < ROWS)

    def runner(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        inv_rms = torch.empty((rows,), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(rows, block_rows),)
        rms_norm_forward_kernel[grid](
            x,
            weight,
            out,
            inv_rms,
            rows,
            cols,
            block_rows,
            block_cols,
            float(eps),
            num_warps=num_warps,
        )
        return out, inv_rms

    return runner


def build_rms_norm_backward_kernel(
    rows: int,
    cols: int,
    block_rows: int,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def rms_norm_backward_kernel(
        x,
        weight,
        inv_rms,
        grad_out,
        grad_x,
        grad_weight_partial,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        row_offsets = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)
        x_values = tl.load(x + row_offsets[:, None] * COLS + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        grad_values = tl.load(
            grad_out + row_offsets[:, None] * COLS + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)
        inv_values = tl.load(inv_rms + row_offsets, mask=row_offsets < ROWS, other=0.0).to(tl.float32)

        weighted_grad = grad_values * weight_values[None, :]
        row_dot = tl.sum(weighted_grad * x_values, axis=1)
        inv_cubed = inv_values * inv_values * inv_values
        grad_x_values = weighted_grad * inv_values[:, None] - x_values * row_dot[:, None] * inv_cubed[:, None] / COLS
        grad_weight_values = grad_values * x_values * inv_values[:, None]
        grad_weight_block = tl.sum(grad_weight_values, axis=0)

        tl.store(grad_x + row_offsets[:, None] * COLS + col_offsets[None, :], grad_x_values, mask=mask)
        tl.store(grad_weight_partial + block_id * COLS + col_offsets, grad_weight_block, mask=col_offsets < COLS)

    def runner(
        x: torch.Tensor,
        weight: torch.Tensor,
        inv_rms: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grad_x = torch.empty_like(x)
        grad_weight_partial = torch.empty(
            (triton.cdiv(rows, block_rows), cols),
            dtype=torch.float32,
            device=x.device,
        )
        grid = (triton.cdiv(rows, block_rows),)
        rms_norm_backward_kernel[grid](
            x,
            weight,
            inv_rms,
            grad_out,
            grad_x,
            grad_weight_partial,
            rows,
            cols,
            block_rows,
            block_cols,
            num_warps=num_warps,
        )
        return grad_x, grad_weight_partial

    return runner


def build_layer_norm_forward_kernel(
    rows: int,
    cols: int,
    block_rows: int,
    eps: float,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def layer_norm_forward_kernel(
        x,
        weight,
        bias,
        out,
        mean,
        inv_std,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)
        col_mask = col_offsets[None, :] < COLS
        x_values = tl.load(x + row_offsets[:, None] * COLS + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)
        bias_values = tl.load(bias + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)

        row_mean = tl.sum(x_values, axis=1) / COLS
        centered = tl.where(col_mask, x_values - row_mean[:, None], 0.0)
        row_inv_std = tl.rsqrt(tl.sum(centered * centered, axis=1) / COLS + EPS)
        out_values = centered * row_inv_std[:, None] * weight_values[None, :] + bias_values[None, :]

        tl.store(out + row_offsets[:, None] * COLS + col_offsets[None, :], out_values, mask=mask)
        tl.store(mean + row_offsets, row_mean, mask=row_offsets < ROWS)
        tl.store(inv_std + row_offsets, row_inv_std, mask=row_offsets < ROWS)

    def runner(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
        inv_std = torch.empty((rows,), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(rows, block_rows),)
        layer_norm_forward_kernel[grid](
            x,
            weight,
            bias,
            out,
            mean,
            inv_std,
            rows,
            cols,
            block_rows,
            block_cols,
            float(eps),
            num_warps=num_warps,
        )
        return out, mean, inv_std

    return runner


def build_layer_norm_inference_kernel(
    rows: int,
    cols: int,
    eps: float,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def layer_norm_inference_kernel(
        x,
        weight,
        bias,
        out,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        EPS: tl.constexpr,
    ):
        row_offset = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = col_offsets < COLS
        x_values = tl.load(x + row_offset * COLS + col_offsets, mask=mask, other=0.0).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=mask, other=0.0).to(tl.float32)
        bias_values = tl.load(bias + col_offsets, mask=mask, other=0.0).to(tl.float32)

        row_mean = tl.sum(x_values, axis=0) / COLS
        centered = tl.where(mask, x_values - row_mean, 0.0)
        row_inv_std = tl.rsqrt(tl.sum(centered * centered, axis=0) / COLS + EPS)
        out_values = centered * row_inv_std * weight_values + bias_values

        tl.store(out + row_offset * COLS + col_offsets, out_values, mask=mask)

    def runner(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        layer_norm_inference_kernel[(rows,)](
            x,
            weight,
            bias,
            out,
            rows,
            cols,
            block_cols,
            float(eps),
            num_warps=num_warps,
        )
        return out

    return runner


def build_layer_norm_backward_kernel(
    rows: int,
    cols: int,
    block_rows: int,
):
    _ensure_triton()
    block_cols = triton_next_power_of_2(cols)
    num_warps = triton_num_warps(block_cols)

    @triton.jit
    def layer_norm_backward_kernel(
        x,
        weight,
        mean,
        inv_std,
        grad_out,
        grad_x,
        grad_weight_partial,
        grad_bias_partial,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        row_offsets = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, BLOCK_COLS)
        mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < COLS)
        col_mask = col_offsets[None, :] < COLS

        x_values = tl.load(x + row_offsets[:, None] * COLS + col_offsets[None, :], mask=mask, other=0.0).to(tl.float32)
        grad_values = tl.load(
            grad_out + row_offsets[:, None] * COLS + col_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(weight + col_offsets, mask=col_offsets < COLS, other=0.0).to(tl.float32)
        mean_values = tl.load(mean + row_offsets, mask=row_offsets < ROWS, other=0.0).to(tl.float32)
        inv_values = tl.load(inv_std + row_offsets, mask=row_offsets < ROWS, other=0.0).to(tl.float32)

        centered = tl.where(col_mask, x_values - mean_values[:, None], 0.0)
        x_hat = centered * inv_values[:, None]
        weighted_grad = grad_values * weight_values[None, :]
        mean_weighted_grad = tl.sum(weighted_grad, axis=1) / COLS
        mean_weighted_grad_xhat = tl.sum(weighted_grad * x_hat, axis=1) / COLS
        grad_x_values = inv_values[:, None] * (
            weighted_grad - mean_weighted_grad[:, None] - x_hat * mean_weighted_grad_xhat[:, None]
        )
        grad_weight_block = tl.sum(grad_values * x_hat, axis=0)
        grad_bias_block = tl.sum(grad_values, axis=0)

        tl.store(grad_x + row_offsets[:, None] * COLS + col_offsets[None, :], grad_x_values, mask=mask)
        tl.store(grad_weight_partial + block_id * COLS + col_offsets, grad_weight_block, mask=col_offsets < COLS)
        tl.store(grad_bias_partial + block_id * COLS + col_offsets, grad_bias_block, mask=col_offsets < COLS)

    def runner(
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_x = torch.empty_like(x)
        partial_shape = (triton.cdiv(rows, block_rows), cols)
        grad_weight_partial = torch.empty(partial_shape, dtype=torch.float32, device=x.device)
        grad_bias_partial = torch.empty(partial_shape, dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(rows, block_rows),)
        layer_norm_backward_kernel[grid](
            x,
            weight,
            mean,
            inv_std,
            grad_out,
            grad_x,
            grad_weight_partial,
            grad_bias_partial,
            rows,
            cols,
            block_rows,
            block_cols,
            num_warps=num_warps,
        )
        return grad_x, grad_weight_partial, grad_bias_partial

    return runner


__all__ = [
    "build_layer_norm_backward_kernel",
    "build_layer_norm_forward_kernel",
    "build_layer_norm_inference_kernel",
    "build_rms_norm_backward_kernel",
    "build_rms_norm_forward_kernel",
]
