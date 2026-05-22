"""LayerNorm operator boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from taac2026.infrastructure.accelerators.tensor_validation import require_cuda_tensors
from taac2026.infrastructure.accelerators.triton_runtime import (
    triton_available,
    triton_supported_floating_dtype,
)
from taac2026.infrastructure.accelerators.normalization.kernels.triton import (
    build_layer_norm_backward_kernel as build_triton_layer_norm_backward_kernel,
    build_layer_norm_forward_kernel as build_triton_layer_norm_forward_kernel,
    build_layer_norm_inference_kernel as build_triton_layer_norm_inference_kernel,
)


LayerNormKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]
LayerNormBackend = Literal["torch", "triton"]

_layer_norm_kernel: LayerNormKernel | None = None


@dataclass(frozen=True, slots=True)
class LayerNormKernelKey:
    rows: int
    cols: int
    dtype: torch.dtype
    eps: float
    block_rows: int


_triton_layer_norm_forward_kernel_cache: dict[
    LayerNormKernelKey, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
] = {}
_triton_layer_norm_inference_kernel_cache: dict[
    LayerNormKernelKey, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
] = {}
_triton_layer_norm_backward_kernel_cache: dict[
    LayerNormKernelKey,
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
] = {}


def clear_layer_norm_kernel_cache() -> None:
    _triton_layer_norm_forward_kernel_cache.clear()
    _triton_layer_norm_inference_kernel_cache.clear()
    _triton_layer_norm_backward_kernel_cache.clear()


def register_layer_norm_kernel(kernel: LayerNormKernel) -> None:
    global _layer_norm_kernel
    _layer_norm_kernel = kernel


def _torch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _normalize_layer_norm_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError("layer_norm expects input with at least 2 dimensions")
    if weight.ndim != 1:
        raise ValueError("layer_norm weight must be a 1D tensor")
    if bias.ndim != 1:
        raise ValueError("layer_norm bias must be a 1D tensor")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(f"last dimension {x.shape[-1]} does not match weight size {weight.shape[0]}")
    if weight.shape != bias.shape:
        raise ValueError(f"layer_norm weight shape {tuple(weight.shape)} does not match bias shape {tuple(bias.shape)}")
    original_shape = x.shape
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    normalized_weight = weight.to(device=matrix.device, dtype=matrix.dtype).contiguous()
    normalized_bias = bias.to(device=matrix.device, dtype=matrix.dtype).contiguous()
    return matrix, normalized_weight, normalized_bias, original_shape


def _resolve_layer_norm_backend(x: torch.Tensor, backend: LayerNormBackend) -> Literal["torch", "triton"]:
    if backend == "torch":
        return "torch"
    if backend != "triton":
        raise ValueError(f"unsupported layer_norm backend: {backend}")
    if not triton_available():
        raise RuntimeError("triton backend requested but triton is not installed")
    require_cuda_tensors("triton layer_norm", x)
    if not triton_supported_floating_dtype(x.dtype):
        raise RuntimeError(f"triton layer_norm does not support dtype {x.dtype}")
    return "triton"


def _layer_norm_registered_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if _layer_norm_kernel is None:
        raise RuntimeError("no registered layer_norm kernel is available")
    return _layer_norm_kernel(x, weight, bias, eps)


def _run_torch_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    if _layer_norm_kernel is not None:
        return _layer_norm_registered_kernel(x, weight, bias, eps)
    return _torch_layer_norm(x, weight, bias, eps)


def _layer_norm_cache_key(x: torch.Tensor, eps: float, block_rows: int | None) -> LayerNormKernelKey:
    return LayerNormKernelKey(
        rows=x.shape[0],
        cols=x.shape[1],
        dtype=x.dtype,
        eps=float(eps),
        block_rows=max(1, int(block_rows or 1)),
    )


def _compile_triton_layer_norm_kernel(key: LayerNormKernelKey) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return _compile_triton_layer_norm_inference_kernel(key)


def _compile_triton_layer_norm_forward_kernel(
    key: LayerNormKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not triton_available():
        raise RuntimeError("triton is not installed")
    if key in _triton_layer_norm_forward_kernel_cache:
        return _triton_layer_norm_forward_kernel_cache[key]

    compiled = build_triton_layer_norm_forward_kernel(key.rows, key.cols, key.block_rows, key.eps)

    def runner(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compiled(x, weight, bias)

    _triton_layer_norm_forward_kernel_cache[key] = runner
    return runner


def _compile_triton_layer_norm_inference_kernel(
    key: LayerNormKernelKey,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if not triton_available():
        raise RuntimeError("triton is not installed")
    if key in _triton_layer_norm_inference_kernel_cache:
        return _triton_layer_norm_inference_kernel_cache[key]

    compiled = build_triton_layer_norm_inference_kernel(key.rows, key.cols, key.eps)

    def runner(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return compiled(x, weight, bias)

    _triton_layer_norm_inference_kernel_cache[key] = runner
    return runner


def _compile_triton_layer_norm_backward_kernel(
    key: LayerNormKernelKey,
) -> Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if not triton_available():
        raise RuntimeError("triton is not installed")
    if key in _triton_layer_norm_backward_kernel_cache:
        return _triton_layer_norm_backward_kernel_cache[key]

    compiled = build_triton_layer_norm_backward_kernel(key.rows, key.cols, key.block_rows)

    def runner(
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        inv_std: torch.Tensor,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compiled(x, weight, mean, inv_std, grad_out)

    _triton_layer_norm_backward_kernel_cache[key] = runner
    return runner


def compile_triton_layer_norm_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    block_rows: int | None = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    matrix, _normalized_weight, _normalized_bias, _original_shape = _normalize_layer_norm_inputs(x, weight, bias)
    key = _layer_norm_cache_key(matrix, eps, block_rows)
    return _compile_triton_layer_norm_kernel(key)


class _TritonLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, block_rows: int) -> torch.Tensor:
        key = _layer_norm_cache_key(x, eps, block_rows)
        forward_kernel = _compile_triton_layer_norm_forward_kernel(key)
        out, mean, inv_std = forward_kernel(x, weight, bias)
        ctx.save_for_backward(x, weight, mean, inv_std)
        ctx.key = key
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, mean, inv_std = ctx.saved_tensors
        backward_kernel = _compile_triton_layer_norm_backward_kernel(ctx.key)
        grad_x, grad_weight_partial, grad_bias_partial = backward_kernel(
            x,
            weight,
            mean,
            inv_std,
            grad_out.contiguous(),
        )
        grad_weight = grad_weight_partial.sum(dim=0).to(weight.dtype)
        grad_bias = grad_bias_partial.sum(dim=0).to(weight.dtype)
        return grad_x, grad_weight, grad_bias, None, None


def _run_triton_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    block_rows: int | None,
) -> torch.Tensor:
    resolved_block_rows = max(1, int(block_rows or 1))
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad or bias.requires_grad):
        return _TritonLayerNormFunction.apply(x, weight, bias, eps, resolved_block_rows)
    kernel = compile_triton_layer_norm_kernel(x, weight, bias, eps, block_rows=resolved_block_rows)
    return kernel(x, weight, bias)


def resolved_layer_norm_backend(
    x: torch.Tensor,
    backend: LayerNormBackend,
    *,
    eps: float = 1e-5,
    block_rows: int | None = None,
) -> Literal["torch", "triton"]:
    del eps, block_rows
    matrix = x.reshape(-1, x.shape[-1]).contiguous() if x.ndim >= 2 else x
    return _resolve_layer_norm_backend(matrix, backend)


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    *,
    backend: LayerNormBackend,
    block_rows: int | None = None,
) -> torch.Tensor:
    matrix, normalized_weight, normalized_bias, original_shape = _normalize_layer_norm_inputs(x, weight, bias)
    resolved_backend = resolved_layer_norm_backend(matrix, backend, eps=eps, block_rows=block_rows)
    if resolved_backend == "torch":
        return _run_torch_layer_norm(matrix, normalized_weight, normalized_bias, eps).reshape(original_shape)
    return _run_triton_layer_norm(
        matrix,
        normalized_weight,
        normalized_bias,
        eps,
        block_rows=block_rows,
    ).reshape(original_shape)


__all__ = [
    "LayerNormBackend",
    "LayerNormKernel",
    "LayerNormKernelKey",
    "clear_layer_norm_kernel_cache",
    "compile_triton_layer_norm_kernel",
    "layer_norm",
    "register_layer_norm_kernel",
    "resolved_layer_norm_backend",
]
