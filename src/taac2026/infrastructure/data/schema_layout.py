"""Compiled PCVR schema layout: physical columns, offsets, vocabularies, lengths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taac2026.domain.schema import PCVRSchema
from taac2026.infrastructure.io.json import read_path


class FeatureSchema:
    """Compiled ``(feature_id, offset, length)`` layout for one feature group."""

    def __init__(self) -> None:
        self.entries: list[tuple[int, int, int]] = []
        self.total_dim: int = 0
        self._fid_to_entry: dict[int, tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> tuple[int, int]:
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> list[int]:
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "total_dim": self.total_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSchema:
        schema = cls()
        for fid, offset, length in payload["entries"]:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = payload["total_dim"]
        return schema


@dataclass(frozen=True, slots=True)
class PCVRSequenceLayout:
    domain: str
    prefix: str
    timestamp_fid: int | None
    feature_ids: tuple[int, ...]
    sideinfo_fids: tuple[int, ...]
    vocab_sizes: dict[int, int]
    max_len: int

    @property
    def sideinfo_vocab_sizes(self) -> list[int]:
        return [self.vocab_sizes[fid] for fid in self.sideinfo_fids]


@dataclass(frozen=True, slots=True)
class PCVRSchemaLayout:
    """Physical compilation of a validated :class:`PCVRSchema`.

    Layout fields are pure physical structure: column tuples, offsets,
    vocabularies and lengths. Model semantics (pair views, time features) are
    never compiled here.
    """

    schema_path: Path
    schema: PCVRSchema
    user_int_cols: tuple[tuple[int, int, int], ...]
    item_int_cols: tuple[tuple[int, int, int], ...]
    user_dense_cols: tuple[tuple[int, int], ...]
    item_dense_cols: tuple[tuple[int, int], ...]
    user_int_schema: FeatureSchema
    item_int_schema: FeatureSchema
    user_dense_schema: FeatureSchema
    item_dense_schema: FeatureSchema
    user_int_vocab_sizes: list[int]
    item_int_vocab_sizes: list[int]
    sequences: dict[str, PCVRSequenceLayout]

    @property
    def seq_domains(self) -> list[str]:
        return sorted(self.sequences)

    @property
    def seq_feature_ids(self) -> dict[str, list[int]]:
        return {domain: list(layout.feature_ids) for domain, layout in self.sequences.items()}

    @property
    def seq_vocab_sizes(self) -> dict[str, dict[int, int]]:
        return {domain: dict(layout.vocab_sizes) for domain, layout in self.sequences.items()}

    @property
    def seq_domain_vocab_sizes(self) -> dict[str, list[int]]:
        return {
            domain: layout.sideinfo_vocab_sizes
            for domain, layout in self.sequences.items()
        }

    @property
    def ts_fids(self) -> dict[str, int | None]:
        return {domain: layout.timestamp_fid for domain, layout in self.sequences.items()}

    @property
    def sideinfo_fids(self) -> dict[str, list[int]]:
        return {domain: list(layout.sideinfo_fids) for domain, layout in self.sequences.items()}

    @property
    def seq_prefix(self) -> dict[str, str]:
        return {domain: layout.prefix for domain, layout in self.sequences.items()}

    @property
    def seq_maxlen(self) -> dict[str, int]:
        return {domain: layout.max_len for domain, layout in self.sequences.items()}

    def required_column_names(self, parquet_schema_names: list[str]) -> tuple[str, ...]:
        available = set(parquet_schema_names)
        names: list[str] = ["timestamp", "label_type", "user_id"]
        names.extend(f"user_int_feats_{fid}" for fid, _vocab_size, _dim in self.user_int_cols)
        names.extend(f"item_int_feats_{fid}" for fid, _vocab_size, _dim in self.item_int_cols)
        names.extend(f"user_dense_feats_{fid}" for fid, _dim in self.user_dense_cols)
        names.extend(f"item_dense_feats_{fid}" for fid, _dim in self.item_dense_cols)
        for layout in self.sequences.values():
            names.extend(f"{layout.prefix}_{fid}" for fid in layout.feature_ids)
        return tuple(dict.fromkeys(name for name in names if name in available))


def _feature_schema_from_int_columns(
    columns: tuple[tuple[int, int, int], ...]
) -> tuple[FeatureSchema, list[int]]:
    schema = FeatureSchema()
    vocab_sizes: list[int] = []
    for fid, vocab_size, dim in columns:
        schema.add(fid, dim)
        vocab_sizes.extend([vocab_size] * dim)
    return schema, vocab_sizes


def _feature_schema_from_dense_columns(
    columns: tuple[tuple[int, int], ...]
) -> FeatureSchema:
    schema = FeatureSchema()
    for fid, dim in columns:
        schema.add(fid, dim)
    return schema


def build_pcvr_schema_layout(
    schema: PCVRSchema,
    seq_max_lens: dict[str, int] | None = None,
    *,
    schema_path: Path | None = None,
) -> PCVRSchemaLayout:
    """Compile a validated :class:`PCVRSchema` into a physical layout."""
    max_lens = seq_max_lens or {}

    user_int_cols = tuple(
        (column.fid, column.vocab_size, column.dim) for column in schema.user_int
    )
    item_int_cols = tuple(
        (column.fid, column.vocab_size, column.dim) for column in schema.item_int
    )
    user_dense_cols = tuple((column.fid, column.dim) for column in schema.user_dense)
    item_dense_cols = tuple((column.fid, column.dim) for column in schema.item_dense)

    user_int_schema, user_int_vocab_sizes = _feature_schema_from_int_columns(user_int_cols)
    item_int_schema, item_int_vocab_sizes = _feature_schema_from_int_columns(item_int_cols)
    user_dense_schema = _feature_schema_from_dense_columns(user_dense_cols)
    item_dense_schema = _feature_schema_from_dense_columns(item_dense_cols)

    sequences: dict[str, PCVRSequenceLayout] = {}
    for domain in sorted(schema.seq):
        config = schema.seq[domain]
        feature_ids = tuple(feature.fid for feature in config.features)
        vocab_sizes = {
            feature.fid: feature.vocab_size for feature in config.features
        }
        sideinfo_fids = tuple(fid for fid in feature_ids if fid != config.ts_fid)
        sequences[domain] = PCVRSequenceLayout(
            domain=domain,
            prefix=config.prefix,
            timestamp_fid=config.ts_fid,
            feature_ids=feature_ids,
            sideinfo_fids=sideinfo_fids,
            vocab_sizes=vocab_sizes,
            max_len=int(max_lens.get(domain, 256)),
        )

    return PCVRSchemaLayout(
        schema_path=schema_path if schema_path is not None else Path(),
        schema=schema,
        user_int_cols=user_int_cols,
        item_int_cols=item_int_cols,
        user_dense_cols=user_dense_cols,
        item_dense_cols=item_dense_cols,
        user_int_schema=user_int_schema,
        item_int_schema=item_int_schema,
        user_dense_schema=user_dense_schema,
        item_dense_schema=item_dense_schema,
        user_int_vocab_sizes=user_int_vocab_sizes,
        item_int_vocab_sizes=item_int_vocab_sizes,
        sequences=sequences,
    )


def load_pcvr_schema_layout(
    schema_path: str | Path,
    seq_max_lens: dict[str, int] | None = None,
) -> PCVRSchemaLayout:
    """Read, validate and compile ``schema.json`` into a physical layout."""
    resolved_path = Path(schema_path).expanduser().resolve()
    raw = read_path(resolved_path)
    schema = PCVRSchema.model_validate(raw)
    return build_pcvr_schema_layout(schema, seq_max_lens, schema_path=resolved_path)
