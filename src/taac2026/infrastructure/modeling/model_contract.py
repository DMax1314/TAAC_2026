"""Shared PCVR schema-driven model construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taac2026.domain.config import PCVRNSConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.infrastructure.data.schema_layout import build_pcvr_schema_layout


def parse_seq_max_lens(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value:
        return result
    for pair in value.split(","):
        if not pair.strip():
            continue
        name, raw_length = pair.split(":", 1)
        result[name.strip()] = int(raw_length.strip())
    return result


def build_feature_specs(schema: Any, per_position_vocab_sizes: list[int]) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    for _feature_id, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset : offset + length])
        specs.append((vocab_size, offset, length))
    return specs


def dataset_schema_path(dataset_path: Path) -> Path:
    """The single schema source for a dataset: inside the dataset dir, else its parent."""
    resolved_dataset_path = dataset_path.expanduser().resolve()
    if resolved_dataset_path.is_dir():
        return resolved_dataset_path / "schema.json"
    return resolved_dataset_path.parent / "schema.json"


def resolve_schema_path(schema_path: Path | None, *, fallback: Path) -> Path:
    """Resolve the authoritative schema path.

    An explicit path must exist (fail fast); otherwise the single ``fallback``
    source is used.
    """
    if schema_path is not None:
        expanded = schema_path.expanduser().resolve()
        if not expanded.exists():
            raise FileNotFoundError(f"schema.json not found at explicit path: {expanded}")
        return expanded
    expanded_fallback = fallback.expanduser().resolve()
    if not expanded_fallback.exists():
        raise FileNotFoundError(f"schema.json not found at fallback path: {expanded_fallback}")
    return expanded_fallback


def resolve_training_schema_path(dataset_path: Path, schema_path: Path | None) -> Path:
    """Training schema source: explicit CLI/env path, else the dataset schema.json."""
    return resolve_schema_path(schema_path, fallback=dataset_schema_path(dataset_path))


def resolve_checkpoint_schema_path(checkpoint_dir: Path, schema_path: Path | None) -> Path:
    """Evaluation/inference schema source: explicit CLI/env path, else the checkpoint schema.json."""
    return resolve_schema_path(schema_path, fallback=checkpoint_dir / "schema.json")


def load_ns_groups(schema: PCVRSchema, ns_config: PCVRNSConfig) -> tuple[list[list[int]], list[list[int]]]:
    """Compile NS token groups from schema entry indices."""
    if ns_config.grouping_strategy == "singleton":
        return (
            [[index] for index in range(len(schema.user_int))],
            [[index] for index in range(len(schema.item_int))],
        )
    user_feature_to_index = {
        column.fid: index for index, column in enumerate(schema.user_int)
    }
    item_feature_to_index = {
        column.fid: index for index, column in enumerate(schema.item_int)
    }
    user_groups = [
        [user_feature_to_index[feature_id] for feature_id in feature_ids]
        for feature_ids in ns_config.user_groups.values()
    ]
    item_groups = [
        [item_feature_to_index[feature_id] for feature_id in feature_ids]
        for feature_ids in ns_config.item_groups.values()
    ]
    return user_groups, item_groups


@dataclass(frozen=True, slots=True)
class PCVRModelSpecs:
    """Compiled schema-derived inputs for uniform model construction."""

    user_int_feature_specs: list[tuple[int, int, int]]
    item_int_feature_specs: list[tuple[int, int, int]]
    user_dense_dim: int
    item_dense_dim: int
    seq_vocab_sizes: dict[str, list[int]]
    user_ns_groups: list[list[int]]
    item_ns_groups: list[list[int]]


def build_pcvr_model_specs(schema: PCVRSchema, ns_config: PCVRNSConfig) -> PCVRModelSpecs:
    """Compile all schema-derived construction inputs for ``model_type(schema, config)``."""
    layout = build_pcvr_schema_layout(schema)
    user_ns_groups, item_ns_groups = load_ns_groups(schema, ns_config)
    return PCVRModelSpecs(
        user_int_feature_specs=build_feature_specs(layout.user_int_schema, layout.user_int_vocab_sizes),
        item_int_feature_specs=build_feature_specs(layout.item_int_schema, layout.item_int_vocab_sizes),
        user_dense_dim=layout.user_dense_schema.total_dim,
        item_dense_dim=layout.item_dense_schema.total_dim,
        seq_vocab_sizes=layout.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
    )


__all__ = [
    "PCVRModelSpecs",
    "build_feature_specs",
    "build_pcvr_model_specs",
    "dataset_schema_path",
    "load_ns_groups",
    "parse_seq_max_lens",
    "resolve_checkpoint_schema_path",
    "resolve_schema_path",
    "resolve_training_schema_path",
]