"""Shared PCVRModelInput factories for experiment contract tests."""

from __future__ import annotations

import torch

from taac2026.infrastructure.data.batches import PCVREntityInput, PCVRModelInput, PCVRSequenceInput


def dualq_contract_model_input() -> PCVRModelInput:
    return PCVRModelInput(
        user=PCVREntityInput(
            int_values=torch.tensor([[1, 11, 12, 1, 14], [2, 0, 0, 2, 0]], dtype=torch.long),
            int_missing_mask=torch.zeros(2, 5, dtype=torch.bool),
            dense_values=torch.randn(2, 6),
            dense_missing_mask=torch.zeros(2, 6, dtype=torch.bool),
        ),
        item=PCVREntityInput(
            int_values=torch.tensor([[1], [2]], dtype=torch.long),
            int_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
            dense_values=torch.randn(2, 3),
            dense_missing_mask=torch.zeros(2, 3, dtype=torch.bool),
        ),
        sequences={
            "seq_a": PCVRSequenceInput(
                values=torch.tensor([[[1, 2, 0, 0]], [[2, 3, 0, 0]]], dtype=torch.long),
                lengths=torch.tensor([2, 2], dtype=torch.long),
                timestamps=torch.tensor([[1000, 2000, 0, 0], [500, 600, 0, 0]], dtype=torch.long),
            ),
            "seq_b": PCVRSequenceInput(
                values=torch.tensor([[[1, 0, 0]], [[2, 3, 0]]], dtype=torch.long),
                lengths=torch.tensor([1, 2], dtype=torch.long),
                timestamps=torch.tensor([[3000, 0, 0], [100, 200, 0]], dtype=torch.long),
            ),
        },
        request_timestamp=torch.tensor([5000, 5000], dtype=torch.long),
    )
