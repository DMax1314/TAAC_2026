"""PCVR persistent schema contract and compiled layout helpers.

``PCVRSchema`` is the validated, immutable, closed contract for ``schema.json``
(raw parquet data only: no pair views, no time-derived features). Compiled
offsets, vocabularies and lengths live in ``infrastructure.data.schema_layout``;
model semantics (pair selection, time features) are model-side.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, field_validator, model_validator

from taac2026.domain.validation import TAACBoundaryModel


class PCVRIntColumn(TAACBoundaryModel):
    """One ``[fid, vocab_size, dim]`` integer feature column."""

    model_config = ConfigDict(frozen=True)

    fid: int
    vocab_size: int
    dim: int

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            if len(value) != 3:
                raise ValueError("int column must be [fid, vocab_size, dim]")
            fid, vocab_size, dim = value
            return {"fid": fid, "vocab_size": vocab_size, "dim": dim}
        return value


class PCVRDenseColumn(TAACBoundaryModel):
    """One ``[fid, dim]`` dense feature column."""

    model_config = ConfigDict(frozen=True)

    fid: int
    dim: int

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError("dense column must be [fid, dim]")
            fid, dim = value
            return {"fid": fid, "dim": dim}
        return value


class PCVRSequenceFeature(TAACBoundaryModel):
    """One ``[fid, vocab_size]`` sequence feature."""

    model_config = ConfigDict(frozen=True)

    fid: int
    vocab_size: int

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError("sequence feature must be [fid, vocab_size]")
            fid, vocab_size = value
            return {"fid": fid, "vocab_size": vocab_size}
        return value


class PCVRSequenceSpec(TAACBoundaryModel):
    """One sequence domain: column prefix, timestamp fid and features."""

    model_config = ConfigDict(frozen=True)

    prefix: str
    ts_fid: int
    features: tuple[PCVRSequenceFeature, ...]


class PCVRSchema(TAACBoundaryModel):
    """Validated, immutable, closed ``schema.json`` contract (raw parquet data)."""

    model_config = ConfigDict(frozen=True)

    format: Literal["raw_parquet"]
    user_int: tuple[PCVRIntColumn, ...]
    item_int: tuple[PCVRIntColumn, ...]
    user_dense: tuple[PCVRDenseColumn, ...]
    item_dense: tuple[PCVRDenseColumn, ...] = ()
    seq: dict[str, PCVRSequenceSpec]

    @field_validator("user_int", "item_int")
    @classmethod
    def _int_columns_non_empty(cls, value: tuple[PCVRIntColumn, ...]) -> tuple[PCVRIntColumn, ...]:
        if not value:
            raise ValueError("int feature columns must not be empty")
        return value

    @model_validator(mode="after")
    def _validate_schema(self) -> PCVRSchema:
        for group_name, columns in (
            ("user_int", self.user_int),
            ("item_int", self.item_int),
        ):
            self._validate_int_columns(group_name, columns)
        for group_name, columns in (
            ("user_dense", self.user_dense),
            ("item_dense", self.item_dense),
        ):
            self._validate_dense_columns(group_name, columns)
        self._validate_sequences(self.seq)
        return self

    @staticmethod
    def _validate_int_columns(
        group_name: str,
        columns: tuple[PCVRIntColumn, ...],
    ) -> None:
        seen: set[int] = set()
        for column in columns:
            if column.fid in seen:
                raise ValueError(f"{group_name} has duplicate fid {column.fid}")
            seen.add(column.fid)
            if column.dim < 1:
                raise ValueError(f"{group_name} fid {column.fid} has dim < 1")
            if column.vocab_size < 0:
                raise ValueError(f"{group_name} fid {column.fid} has negative vocab_size")

    @staticmethod
    def _validate_dense_columns(
        group_name: str,
        columns: tuple[PCVRDenseColumn, ...],
    ) -> None:
        seen: set[int] = set()
        for column in columns:
            if column.fid in seen:
                raise ValueError(f"{group_name} has duplicate fid {column.fid}")
            seen.add(column.fid)
            if column.dim < 1:
                raise ValueError(f"{group_name} fid {column.fid} has dim < 1")

    @staticmethod
    def _validate_sequences(sequences: dict[str, PCVRSequenceSpec]) -> None:
        seen_prefixes: set[str] = set()
        for domain, spec in sequences.items():
            if not domain:
                raise ValueError("sequence domain name must not be empty")
            if not spec.prefix:
                raise ValueError(f"sequence {domain!r} has empty prefix")
            if spec.prefix in seen_prefixes:
                raise ValueError(f"duplicate sequence prefix {spec.prefix!r}")
            seen_prefixes.add(spec.prefix)
            if not spec.features:
                raise ValueError(f"sequence {domain!r} has no features")
            feature_fids = [feature.fid for feature in spec.features]
            if len(feature_fids) != len(set(feature_fids)):
                raise ValueError(f"sequence {domain!r} has duplicate feature fids")
            if spec.ts_fid not in feature_fids:
                raise ValueError(
                    f"sequence {domain!r} ts_fid {spec.ts_fid} is not a feature fid"
                )
            for feature in spec.features:
                if feature.vocab_size < 0:
                    raise ValueError(
                        f"sequence {domain!r} fid {feature.fid} has negative vocab_size"
                    )


__all__ = [
    "PCVRDenseColumn",
    "PCVRIntColumn",
    "PCVRSchema",
    "PCVRSequenceFeature",
    "PCVRSequenceSpec",
]
