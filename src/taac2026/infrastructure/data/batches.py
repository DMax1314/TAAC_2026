"""Structured PCVR batch carriers and tensor utilities.

``PCVRBatch`` is the canonical data contract shared by the data pipeline,
trainer, evaluation and every experiment model. It carries raw values, masks
and timestamps only; time features (buckets, stats) are model-side derived.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass(frozen=True, slots=True)
class PCVRSequenceInput:
    """One sequence domain: values, lengths and raw event timestamps."""

    values: torch.Tensor  # [B, F, L] int64
    lengths: torch.Tensor  # [B] int64
    timestamps: torch.Tensor  # [B, L] int64 raw event timestamps

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PCVRSequenceInput:
        return PCVRSequenceInput(
            values=self.values.to(device, non_blocking=non_blocking),
            lengths=self.lengths.to(device, non_blocking=non_blocking),
            timestamps=self.timestamps.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class PCVREntityInput:
    """User or item entity: int/dense values plus missing masks."""

    int_values: torch.Tensor  # [B, D] int64
    int_missing_mask: torch.Tensor  # [B, D] bool
    dense_values: torch.Tensor  # [B, Dd] float32
    dense_missing_mask: torch.Tensor  # [B, Dd] bool

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PCVREntityInput:
        return PCVREntityInput(
            int_values=self.int_values.to(device, non_blocking=non_blocking),
            int_missing_mask=self.int_missing_mask.to(device, non_blocking=non_blocking),
            dense_values=self.dense_values.to(device, non_blocking=non_blocking),
            dense_missing_mask=self.dense_missing_mask.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class PCVRModelInput:
    """Unified model-facing input consumed by every experiment model."""

    user: PCVREntityInput
    item: PCVREntityInput
    sequences: dict[str, PCVRSequenceInput]
    request_timestamp: torch.Tensor  # [B] int64

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PCVRModelInput:
        return PCVRModelInput(
            user=self.user.to(device, non_blocking=non_blocking),
            item=self.item.to(device, non_blocking=non_blocking),
            sequences={
                domain: sequence.to(device, non_blocking=non_blocking)
                for domain, sequence in self.sequences.items()
            },
            request_timestamp=self.request_timestamp.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class PCVRBatch:
    """Canonical converted batch produced by the data pipeline."""

    inputs: PCVRModelInput
    label: torch.Tensor  # [B] int64
    user_id: list[Any]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> PCVRBatch:
        return PCVRBatch(
            inputs=self.inputs.to(device, non_blocking=non_blocking),
            label=self.label.to(device, non_blocking=non_blocking),
            user_id=self.user_id,
        )


PCVRBatchFactory = Callable[[], PCVRBatch]


@dataclass(frozen=True, slots=True)
class PCVRSharedTensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


class PCVRBatchTransform(Protocol):
    def __call__(self, batch: PCVRBatch, *, generator: torch.Generator) -> PCVRBatch:
        """Return a transformed PCVR batch."""


def pcvr_batch_row_count(batch: PCVRBatch) -> int:
    if batch.label.ndim > 0:
        return int(batch.label.shape[0])
    return int(batch.inputs.request_timestamp.shape[0])


def pcvr_tensor_paths(seq_domains: list[str]) -> dict[str, tuple[str, ...]]:
    """Flat cache key to structured tensor path for every batch tensor."""
    paths: dict[str, tuple[str, ...]] = {
        "user_int_values": ("inputs", "user", "int_values"),
        "user_int_missing_mask": ("inputs", "user", "int_missing_mask"),
        "user_dense_values": ("inputs", "user", "dense_values"),
        "user_dense_missing_mask": ("inputs", "user", "dense_missing_mask"),
        "item_int_values": ("inputs", "item", "int_values"),
        "item_int_missing_mask": ("inputs", "item", "int_missing_mask"),
        "item_dense_values": ("inputs", "item", "dense_values"),
        "item_dense_missing_mask": ("inputs", "item", "dense_missing_mask"),
        "request_timestamp": ("inputs", "request_timestamp"),
        "label": ("label",),
    }
    for domain in seq_domains:
        paths[f"{domain}_values"] = ("inputs", "sequences", domain, "values")
        paths[f"{domain}_lengths"] = ("inputs", "sequences", domain, "lengths")
        paths[f"{domain}_timestamps"] = ("inputs", "sequences", domain, "timestamps")
    return paths


def get_pcvr_batch_tensor(batch: PCVRBatch, path: tuple[str, ...]) -> torch.Tensor:
    value: Any = batch
    for part in path:
        if isinstance(value, dict):
            value = value[part]
        else:
            value = getattr(value, part)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch path {path!r} does not resolve to a tensor")
    return value


def pcvr_batch_from_parts(
    tensors: dict[str, torch.Tensor],
    *,
    seq_domains: list[str],
    user_id: list[Any],
) -> PCVRBatch:
    """Rebuild a structured batch from flat cache parts."""
    user = PCVREntityInput(
        int_values=tensors["user_int_values"],
        int_missing_mask=tensors["user_int_missing_mask"],
        dense_values=tensors["user_dense_values"],
        dense_missing_mask=tensors["user_dense_missing_mask"],
    )
    item = PCVREntityInput(
        int_values=tensors["item_int_values"],
        int_missing_mask=tensors["item_int_missing_mask"],
        dense_values=tensors["item_dense_values"],
        dense_missing_mask=tensors["item_dense_missing_mask"],
    )
    sequences = {
        domain: PCVRSequenceInput(
            values=tensors[f"{domain}_values"],
            lengths=tensors[f"{domain}_lengths"],
            timestamps=tensors[f"{domain}_timestamps"],
        )
        for domain in seq_domains
    }
    inputs = PCVRModelInput(
        user=user,
        item=item,
        sequences=sequences,
        request_timestamp=tensors["request_timestamp"],
    )
    return PCVRBatch(inputs=inputs, label=tensors["label"], user_id=list(user_id or []))


def _sequence_domains_of(batch: PCVRBatch) -> list[str]:
    return list(batch.inputs.sequences)


def _entity_tensors(batch: PCVRBatch) -> dict[str, torch.Tensor]:
    return {
        "user_int_values": batch.inputs.user.int_values,
        "user_int_missing_mask": batch.inputs.user.int_missing_mask,
        "user_dense_values": batch.inputs.user.dense_values,
        "user_dense_missing_mask": batch.inputs.user.dense_missing_mask,
        "item_int_values": batch.inputs.item.int_values,
        "item_int_missing_mask": batch.inputs.item.int_missing_mask,
        "item_dense_values": batch.inputs.item.dense_values,
        "item_dense_missing_mask": batch.inputs.item.dense_missing_mask,
    }


def _tensor_for(batch: PCVRBatch, key: str) -> torch.Tensor:
    entity_tensors = _entity_tensors(batch)
    if key in entity_tensors:
        return entity_tensors[key]
    if key == "request_timestamp":
        return batch.inputs.request_timestamp
    if key == "label":
        return batch.label
    suffix_map = {"_values": "values", "_lengths": "lengths", "_timestamps": "timestamps"}
    for suffix, field in suffix_map.items():
        if key.endswith(suffix):
            domain = key[: -len(suffix)]
            return getattr(batch.inputs.sequences[domain], field)
    raise KeyError(key)


def clone_pcvr_batch(batch: PCVRBatch) -> PCVRBatch:
    return PCVRBatch(
        inputs=PCVRModelInput(
            user=PCVREntityInput(
                int_values=batch.inputs.user.int_values.clone(),
                int_missing_mask=batch.inputs.user.int_missing_mask.clone(),
                dense_values=batch.inputs.user.dense_values.clone(),
                dense_missing_mask=batch.inputs.user.dense_missing_mask.clone(),
            ),
            item=PCVREntityInput(
                int_values=batch.inputs.item.int_values.clone(),
                int_missing_mask=batch.inputs.item.int_missing_mask.clone(),
                dense_values=batch.inputs.item.dense_values.clone(),
                dense_missing_mask=batch.inputs.item.dense_missing_mask.clone(),
            ),
            sequences={
                domain: PCVRSequenceInput(
                    values=sequence.values.clone(),
                    lengths=sequence.lengths.clone(),
                    timestamps=sequence.timestamps.clone(),
                )
                for domain, sequence in batch.inputs.sequences.items()
            },
            request_timestamp=batch.inputs.request_timestamp.clone(),
        ),
        label=batch.label.clone(),
        user_id=list(batch.user_id),
    )


def repeat_pcvr_rows(batch: PCVRBatch, repeats: int) -> PCVRBatch:
    if repeats <= 1:
        return clone_pcvr_batch(batch)
    row_count = pcvr_batch_row_count(batch)

    def repeat_tensor(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 0 and value.shape[0] == row_count:
            return value.repeat_interleave(repeats, dim=0)
        return value

    return PCVRBatch(
        inputs=PCVRModelInput(
            user=PCVREntityInput(
                int_values=repeat_tensor(batch.inputs.user.int_values),
                int_missing_mask=repeat_tensor(batch.inputs.user.int_missing_mask),
                dense_values=repeat_tensor(batch.inputs.user.dense_values),
                dense_missing_mask=repeat_tensor(batch.inputs.user.dense_missing_mask),
            ),
            item=PCVREntityInput(
                int_values=repeat_tensor(batch.inputs.item.int_values),
                int_missing_mask=repeat_tensor(batch.inputs.item.int_missing_mask),
                dense_values=repeat_tensor(batch.inputs.item.dense_values),
                dense_missing_mask=repeat_tensor(batch.inputs.item.dense_missing_mask),
            ),
            sequences={
                domain: PCVRSequenceInput(
                    values=repeat_tensor(sequence.values),
                    lengths=repeat_tensor(sequence.lengths),
                    timestamps=repeat_tensor(sequence.timestamps),
                )
                for domain, sequence in batch.inputs.sequences.items()
            },
            request_timestamp=repeat_tensor(batch.inputs.request_timestamp),
        ),
        label=repeat_tensor(batch.label),
        user_id=[item for item in batch.user_id for _repeat_index in range(repeats)],
    )


def take_pcvr_rows(batch: PCVRBatch, row_indices: torch.Tensor) -> PCVRBatch:
    indices = row_indices.detach().cpu().tolist()
    row_count = pcvr_batch_row_count(batch)

    def take_tensor(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 0 and value.shape[0] == row_count:
            return value.index_select(0, row_indices.to(value.device))
        return value

    return PCVRBatch(
        inputs=PCVRModelInput(
            user=PCVREntityInput(
                int_values=take_tensor(batch.inputs.user.int_values),
                int_missing_mask=take_tensor(batch.inputs.user.int_missing_mask),
                dense_values=take_tensor(batch.inputs.user.dense_values),
                dense_missing_mask=take_tensor(batch.inputs.user.dense_missing_mask),
            ),
            item=PCVREntityInput(
                int_values=take_tensor(batch.inputs.item.int_values),
                int_missing_mask=take_tensor(batch.inputs.item.int_missing_mask),
                dense_values=take_tensor(batch.inputs.item.dense_values),
                dense_missing_mask=take_tensor(batch.inputs.item.dense_missing_mask),
            ),
            sequences={
                domain: PCVRSequenceInput(
                    values=take_tensor(sequence.values),
                    lengths=take_tensor(sequence.lengths),
                    timestamps=take_tensor(sequence.timestamps),
                )
                for domain, sequence in batch.inputs.sequences.items()
            },
            request_timestamp=take_tensor(batch.inputs.request_timestamp),
        ),
        label=take_tensor(batch.label),
        user_id=(
            [batch.user_id[index] for index in indices]
            if len(batch.user_id) == row_count
            else list(batch.user_id)
        ),
    )


def concat_pcvr_batches(batches: list[PCVRBatch]) -> PCVRBatch:
    if not batches:
        raise ValueError("cannot concat an empty list of batches")
    first = batches[0]
    domains = _sequence_domains_of(first)

    def concat_tensor(key: str) -> torch.Tensor:
        return torch.cat([_tensor_for(batch, key) for batch in batches], dim=0)

    user = PCVREntityInput(
        int_values=concat_tensor("user_int_values"),
        int_missing_mask=concat_tensor("user_int_missing_mask"),
        dense_values=concat_tensor("user_dense_values"),
        dense_missing_mask=concat_tensor("user_dense_missing_mask"),
    )
    item = PCVREntityInput(
        int_values=concat_tensor("item_int_values"),
        int_missing_mask=concat_tensor("item_int_missing_mask"),
        dense_values=concat_tensor("item_dense_values"),
        dense_missing_mask=concat_tensor("item_dense_missing_mask"),
    )
    sequences = {
        domain: PCVRSequenceInput(
            values=concat_tensor(f"{domain}_values"),
            lengths=concat_tensor(f"{domain}_lengths"),
            timestamps=concat_tensor(f"{domain}_timestamps"),
        )
        for domain in domains
    }
    inputs = PCVRModelInput(
        user=user,
        item=item,
        sequences=sequences,
        request_timestamp=concat_tensor("request_timestamp"),
    )
    user_ids: list[Any] = []
    for batch in batches:
        user_ids.extend(batch.user_id)
    return PCVRBatch(inputs=inputs, label=concat_tensor("label"), user_id=user_ids)


__all__ = [
    "PCVRBatch",
    "PCVRBatchFactory",
    "PCVRBatchTransform",
    "PCVREntityInput",
    "PCVRModelInput",
    "PCVRSequenceInput",
    "PCVRSharedTensorSpec",
    "clone_pcvr_batch",
    "concat_pcvr_batches",
    "get_pcvr_batch_tensor",
    "pcvr_batch_from_parts",
    "pcvr_batch_row_count",
    "pcvr_tensor_paths",
    "repeat_pcvr_rows",
    "take_pcvr_rows",
]
