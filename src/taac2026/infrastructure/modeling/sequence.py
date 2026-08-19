"""Sequence modeling helpers."""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Literal

import torch
from torch.utils.checkpoint import checkpoint

from taac2026.infrastructure.accelerators.attention.flash_attention import flash_attention


FlashAttentionBackend = Literal["torch", "tilelang"]
_FLASH_ATTENTION_BACKEND: ContextVar[FlashAttentionBackend] = ContextVar(
    "taac2026_flash_attention_backend",
    default="torch",
)


def configure_flash_attention_runtime(*, backend: str) -> None:
    if backend not in {"torch", "tilelang"}:
        raise ValueError(f"unsupported flash attention backend: {backend}")
    _FLASH_ATTENTION_BACKEND.set(backend)


def flash_attention_runtime_state() -> FlashAttentionBackend:
    return _FLASH_ATTENTION_BACKEND.get()


def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return mask
    all_masked = mask.all(dim=1, keepdim=True)
    if mask.shape[1] == 0:
        return mask
    first_column = torch.zeros_like(mask)
    first_column[:, :1] = True
    return torch.where(all_masked, mask & ~first_column, mask)


def maybe_gradient_checkpoint(function, *args, enabled: bool = False, **kwargs):
    if not enabled or not torch.is_grad_enabled():
        return function(*args, **kwargs)
    return checkpoint(function, *args, use_reentrant=False, **kwargs)


def masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    if padding_mask is None:
        return tokens.mean(dim=1)
    valid = (~padding_mask).to(tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def masked_last(tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if tokens.shape[1] == 0:
        return tokens.new_zeros(tokens.shape[0], tokens.shape[-1])
    indices = lengths.clamp_min(1).clamp_max(tokens.shape[1]).to(torch.long) - 1
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
    return tokens[batch_indices, indices]


def choose_num_heads(d_model: int, requested_heads: int) -> int:
    requested_heads = max(1, requested_heads)
    if d_model % requested_heads == 0:
        return requested_heads
    for heads in range(min(requested_heads, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
    backend: FlashAttentionBackend | None = None,
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    threads: int = 128,
) -> torch.Tensor:
    batch_size, query_len, d_model = q.shape
    head_dim = d_model // num_heads
    q = q.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, k.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, v.shape[1], num_heads, head_dim).transpose(1, 2)
    output = flash_attention(
        q,
        k,
        v,
        backend=flash_attention_runtime_state() if backend is None else backend,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        training=training,
        is_causal=is_causal,
        block_m=block_m,
        block_n=block_n,
        num_stages=num_stages,
        threads=threads,
    )
    return output.transpose(1, 2).contiguous().view(batch_size, query_len, d_model)


def causal_valid_attention_mask(padding_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, token_count = padding_mask.shape
    causal = torch.ones(token_count, token_count, dtype=torch.bool, device=padding_mask.device).tril()
    key_valid = ~padding_mask
    mask = causal.unsqueeze(0) & key_valid.unsqueeze(1)
    query_invalid = padding_mask.unsqueeze(-1)
    fallback = torch.eye(token_count, dtype=torch.bool, device=padding_mask.device).unsqueeze(0)
    mask = torch.where(query_invalid, fallback, mask)
    return mask.unsqueeze(1).expand(batch_size, num_heads, token_count, token_count)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if length == 0:
        return torch.empty(0, dim, device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    values = torch.zeros(length, dim, device=device)
    values[:, 0::2] = torch.sin(positions * frequencies)
    values[:, 1::2] = torch.cos(positions * frequencies[: values[:, 1::2].shape[1]])
    return values


def deduplicate_sequence_events(
    values: torch.Tensor,
    lengths: torch.Tensor,
    timestamps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop duplicate event signatures per row, keeping the last occurrence.

    Returns (values, lengths, timestamps) compacted in original event order.
    Rows with raw length <= 1 are returned unchanged; changed rows are zero-filled
    past their new length. Event signatures use the sequence feature values only,
    so timestamps follow the kept events without participating in the signature.
    """
    batch_size, feature_count, max_len = values.shape
    raw_lengths = torch.clamp(lengths, min=0, max=max_len).to(torch.long)
    positions = torch.arange(max_len, device=values.device)
    active = (values > 0).any(dim=1) & (positions[None, :] < raw_lengths[:, None])

    events = values.transpose(1, 2)  # [B, L, F]
    masked = torch.where(active.unsqueeze(-1), events, torch.full_like(events, -1))
    order = positions.expand(batch_size, -1).clone()
    for feature_index in range(feature_count - 1, -1, -1):
        keys = masked.gather(1, order.unsqueeze(-1).expand(-1, -1, feature_count))[:, :, feature_index]
        permutation = torch.argsort(keys, dim=1, stable=True)
        order = order.gather(1, permutation)
    sorted_masked = masked.gather(1, order.unsqueeze(-1).expand(-1, -1, feature_count))
    sorted_active = active.gather(1, order)
    last_of_group = torch.cat(
        [
            (sorted_masked[:, 1:] != sorted_masked[:, :-1]).any(dim=-1),
            torch.ones(batch_size, 1, dtype=torch.bool, device=values.device),
        ],
        dim=1,
    )
    keep_sorted = last_of_group & sorted_active
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(1, order, keep_sorted)

    keep_count = keep.sum(dim=1)
    changed_rows = (keep_count != raw_lengths) & (raw_lengths > 1)
    ranks = torch.cumsum(keep.to(torch.long), dim=1) - 1
    flat_keep = keep.flatten()
    flat_rank = ranks.flatten()[flat_keep]
    row_ids = torch.arange(batch_size, device=values.device).repeat_interleave(max_len)[flat_keep]
    src_positions = positions.expand(batch_size, -1).flatten()[flat_keep]
    gather_flat = torch.zeros(batch_size * max_len, dtype=torch.long, device=values.device)
    gather_flat.scatter_(0, row_ids * max_len + flat_rank, src_positions)
    gather_index = gather_flat.view(batch_size, max_len)

    gathered_values = values.gather(2, gather_index.unsqueeze(1).expand(-1, feature_count, -1))
    gathered_timestamps = timestamps.gather(1, gather_index)
    zero_out = (positions[None, :] >= keep_count[:, None]) & changed_rows[:, None]
    gathered_values = gathered_values.masked_fill(zero_out.unsqueeze(1), 0)
    gathered_timestamps = gathered_timestamps.masked_fill(zero_out, 0)

    new_values = torch.where(changed_rows[:, None, None], gathered_values, values)
    new_timestamps = torch.where(changed_rows[:, None], gathered_timestamps, timestamps)
    new_lengths = torch.where(changed_rows, keep_count.to(lengths.dtype), lengths)
    return new_values, new_lengths, new_timestamps


__all__ = [
    "FlashAttentionBackend",
    "causal_valid_attention_mask",
    "choose_num_heads",
    "configure_flash_attention_runtime",
    "deduplicate_sequence_events",
    "flash_attention_runtime_state",
    "make_padding_mask",
    "masked_last",
    "masked_mean",
    "maybe_gradient_checkpoint",
    "safe_key_padding_mask",
    "scaled_dot_product_attention",
    "sinusoidal_positions",
]
