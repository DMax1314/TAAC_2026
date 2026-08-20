"""DualQ typed configuration.

All model switches live in :class:`DualQModelConfig` (a
``PCVRModelConfig`` subclass) so they are parsed by the shared CLI, written
into the checkpoint sidecar, and rebuilt from it by the shared runtime.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator

from taac2026.api import PCVRModelConfig, PCVRTrainConfig

from .layers import parse_pair_feature_fids


class DualQModelConfig(PCVRModelConfig):
    """DualQ-specific model switches (source run.sh flags)."""

    model_config = ConfigDict(frozen=True)

    use_global_time_token: bool = True
    use_seq_gap_buckets: bool = True
    use_time_gap_domain_gates: bool = True
    use_fid87_token_residual: bool = True
    use_time_decay_summary: bool = True
    compress_high_cardinality: bool = True
    use_fm_highway: bool = True
    # DualQ query budget: must equal user_q_tokens + item_q_tokens.
    num_queries: int = 6
    user_q_tokens: int = 4
    item_q_tokens: int = 2
    use_time_aligned_interleave: bool = True
    seq_interest_ratios: str = "1.0,0.7"
    pair_feature_fids: str = "62,63,64,65,66,89,90,91"
    # User dense split: fid=61 (user embedding) | fid=87 (history seq blocks).
    user_emb_dim: int = 256
    user_seq_block_dim: int = 32
    user_seq_num: int = 10

    @field_validator("user_q_tokens", "item_q_tokens")
    @classmethod
    def _validate_q_tokens_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"query token counts must be positive, got {value}")
        return value

    @field_validator("pair_feature_fids")
    @classmethod
    def _validate_pair_feature_fids(cls, value: str) -> str:
        fids = parse_pair_feature_fids(value)
        if len(set(fids)) != len(fids):
            raise ValueError(f"duplicate pair fids: {value}")
        return value

    @property
    def seq_interest_ratio_list(self) -> list[float]:
        text = self.seq_interest_ratios.strip()
        if not text:
            return [1.0]
        ratios = [float(part) for part in text.split(",") if part.strip()]
        if not ratios or any(ratio <= 0.0 for ratio in ratios):
            raise ValueError(f"seq_interest_ratios must be positive floats, got {self.seq_interest_ratios!r}")
        return ratios


class DualQTrainConfig(PCVRTrainConfig):
    """Full typed training config for the dualq experiment."""

    model: DualQModelConfig = Field(default_factory=DualQModelConfig)


__all__ = ["DualQModelConfig", "DualQTrainConfig"]
