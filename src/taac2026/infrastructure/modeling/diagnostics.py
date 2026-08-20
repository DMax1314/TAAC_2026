"""Representation diagnostics shared by PCVR experiment models."""

from __future__ import annotations

import torch


def masked_effective_rank(
    tokens: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the mean entropy-based effective rank of batched token matrices.

    Padding positions are excluded before centering. Samples without any
    singular-value mass contribute zero instead of the misleading ``exp(0)``.
    """
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [batch, tokens, hidden], got {tuple(tokens.shape)}")
    if padding_mask is None:
        padding_mask = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    if padding_mask.shape != tokens.shape[:2]:
        raise ValueError(
            "padding_mask must match the first two token dimensions: "
            f"expected {tuple(tokens.shape[:2])}, got {tuple(padding_mask.shape)}"
        )
    if tokens.shape[0] == 0 or tokens.shape[1] == 0 or tokens.shape[2] == 0:
        return tokens.new_zeros((), dtype=torch.float32)

    with torch.no_grad():
        hidden = tokens.detach().float()
        valid = (~padding_mask.to(device=tokens.device, dtype=torch.bool)).unsqueeze(-1)
        valid_float = valid.to(hidden.dtype)
        centered = hidden - (hidden * valid_float).sum(dim=1, keepdim=True) / valid_float.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        centered = centered * valid_float
        singular_values = torch.linalg.svdvals(centered)
        mass = singular_values.sum(dim=-1, keepdim=True)
        weights = singular_values / mass.clamp_min(1.0e-12)
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=-1)
        effective_rank = torch.exp(entropy)
        effective_rank = torch.where(mass.squeeze(-1) > 1.0e-12, effective_rank, torch.zeros_like(effective_rank))
        return effective_rank.mean()


__all__ = ["masked_effective_rank"]
