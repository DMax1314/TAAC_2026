"""Symbiosis V2/V3 configuration surface."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from taac2026.api import PCVRModelConfig, PCVRTrainConfig


class SymbiosisModelConfig(PCVRModelConfig):
    """PCVR model config extended with Symbiosis V2/V3 options."""

    model_config = ConfigDict(frozen=True)

    v2_use_dense_tokens: bool = True
    v2_use_missing_tokens: bool = True
    v2_use_sequence_stats_tokens: bool = True
    v2_use_metadata_attention_bias: bool = True
    v2_use_candidate_readout: bool = True
    v2_tokenization_mode: str = "group"
    v2_sparse_seed: int = 20260512
    v2_recent_event_tokens: int = 16
    v2_memory_event_tokens: int = 8
    v2_user_dense_tokens: int = 3
    v2_item_dense_tokens: int = 1
    v2_user_missing_tokens: int = 2
    v2_item_missing_tokens: int = 1
    v2_high_risk_token_dropout_rate: float = 0.08
    v2_compress_large_ids: bool = True
    v2_compile_backbone: bool = True
    v3_enabled: bool = True
    v3_memory_selection_mode: str = "quality_stratified"
    v3_recent_event_tokens_by_domain: str = "seq_a:8,seq_b:8,seq_c:20,seq_d:24"
    v3_memory_event_tokens_by_domain: str = "seq_a:4,seq_b:4,seq_c:10,seq_d:12"
    v3_memory_density_weight: float = 1.0
    v3_memory_time_weight: float = 0.30
    v3_memory_recency_weight: float = 0.20
    v3_memory_duplicate_penalty: float = 0.50


class SymbiosisTrainConfig(PCVRTrainConfig):
    """Full typed training configuration with the Symbiosis model config."""

    model: SymbiosisModelConfig = Field(default_factory=SymbiosisModelConfig)


__all__ = [
    "SymbiosisModelConfig",
    "SymbiosisTrainConfig",
]
