"""Typed configuration for the QueryFormer experiment."""

from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator

from taac2026.api import PCVRModelConfig, PCVRTrainConfig


class QueryFormerModelConfig(PCVRModelConfig):
    """QueryFormer architecture and ablation switches."""

    model_config = ConfigDict(frozen=True)

    num_embedding_columns: int = 4
    dcn_num_layers: int = 2
    compress_high_cardinality: bool = True
    use_query_self_attention: bool = True
    use_query_cross_attention: bool = True
    use_query_seq_cross_attention: bool = True
    use_seq_query_cross_attention: bool = True

    @field_validator("num_embedding_columns")
    @classmethod
    def _validate_num_embedding_columns(cls, value: int) -> int:
        if not 1 <= value <= 8:
            raise ValueError("num_embedding_columns must be between 1 and 8")
        return value

    @field_validator("dcn_num_layers", "num_queries", "num_blocks")
    @classmethod
    def _validate_positive_counts(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "dcn_num_layers, num_queries, and num_blocks must be positive"
            )
        return value


class QueryFormerTrainConfig(PCVRTrainConfig):
    """Full typed training configuration for QueryFormer."""

    model: QueryFormerModelConfig = Field(default_factory=QueryFormerModelConfig)


__all__ = ["QueryFormerModelConfig", "QueryFormerTrainConfig"]
