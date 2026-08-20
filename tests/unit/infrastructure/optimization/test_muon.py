"""Muon contracts for independently batched matrix parameters."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from taac2026.infrastructure.optimization.muon import (
    Muon,
    mark_muon_adamw,
    mark_muon_batched_matrix,
)


def test_batched_matrix_update_matches_independent_muon_parameters() -> None:
    torch.manual_seed(7)
    initial = torch.randn(3, 4, 5)
    gradient = torch.randn_like(initial)
    batched = mark_muon_batched_matrix(nn.Parameter(initial.clone()))
    independent = [nn.Parameter(matrix.clone()) for matrix in initial]
    batched.grad = gradient.clone()
    for parameter, matrix_gradient in zip(independent, gradient, strict=True):
        parameter.grad = matrix_gradient.clone()

    options = {
        "lr": 0.01,
        "momentum": 0.0,
        "nesterov": False,
        "ns_steps": 3,
        "weight_decay": 0.0,
    }
    Muon([batched], **options).step()
    Muon(independent, **options).step()

    assert torch.allclose(batched, torch.stack(independent))


def test_batched_matrix_marker_rejects_non_matrix_batches() -> None:
    with pytest.raises(
        ValueError, match="batched-matrix parameters must be three-dimensional"
    ):
        mark_muon_batched_matrix(nn.Parameter(torch.empty(2, 3)))


def test_marked_matrix_uses_adamw_like_independent_vectors() -> None:
    torch.manual_seed(11)
    initial = torch.randn(3, 5)
    gradient = torch.randn_like(initial)
    batched = mark_muon_adamw(nn.Parameter(initial.clone()))
    independent = [nn.Parameter(vector.clone()) for vector in initial]
    batched.grad = gradient.clone()
    for parameter, vector_gradient in zip(independent, gradient, strict=True):
        parameter.grad = vector_gradient.clone()

    options = {"lr": 0.01, "weight_decay": 0.02}
    Muon([batched], **options).step()
    Muon(independent, **options).step()

    assert torch.allclose(batched, torch.stack(independent))
