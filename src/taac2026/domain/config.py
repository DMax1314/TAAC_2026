"""Typed configuration objects for PCVR experiment packages.

All configuration is modeled as nested, frozen Pydantic models derived from
``TAACBoundaryModel`` so that CLI parsing (Tyro), checkpoint sidecars, and
experiment defaults share one authoritative representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from taac2026.domain.runtime_config import (
    DENSE_OPTIMIZER_TYPE_CHOICES,
    PCVRLossConfig,
    PCVRLossTermConfig as PCVRLossTermConfig,
    RuntimeExecutionConfig,
)
from taac2026.domain.validation import TAACBoundaryModel


SeqEncoderType = Literal["swiglu", "transformer", "longer"]
RankMixerMode = Literal["full", "ffn_only", "none"]
NSTokenizerType = Literal["group", "rankmixer"]
NSGroupingStrategy = Literal["explicit", "singleton"]
PCVRSeqWindowMode = Literal["tail", "random_tail", "rolling"]
PCVRDataCacheMode = Literal["none", "lru", "fifo", "lfu", "rr", "opt"]
PCVRDataSplitStrategy = Literal["row_group_tail", "timestamp_auto", "user_hash", "sample_hash"]
PCVRDataSamplingStrategy = Literal["step_random", "row_group_sweep"]
DenseOptimizerType = Literal["adamw", "fused_adamw", "orthogonal_adamw", "muon"]
DenseLRSchedulerType = Literal["none", "linear", "cosine"]
PCVREarlyStoppingMetric = Literal["auc", "logloss"]
RMSNormBackend = Literal["torch", "tilelang", "triton"]
FlashAttentionBackend = Literal["torch", "tilelang"]
RMS_NORM_BACKEND_CHOICES = ("torch", "tilelang", "triton")
FLASH_ATTENTION_BACKEND_CHOICES = ("torch", "tilelang")


DENSE_LR_SCHEDULER_TYPE_CHOICES = ("none", "linear", "cosine")
PCVR_DATA_CACHE_MODE_CHOICES = ("none", "lru", "fifo", "lfu", "rr", "opt")
PCVR_DATA_SPLIT_STRATEGY_CHOICES = ("row_group_tail", "timestamp_auto", "user_hash", "sample_hash")
PCVR_DATA_SAMPLING_STRATEGY_CHOICES = ("step_random", "row_group_sweep")
PCVR_EARLY_STOPPING_METRIC_CHOICES = ("auc", "logloss")


def _normalize_ns_group_map(groups: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    return {
        str(group_name): [int(feature_id) for feature_id in feature_ids]
        for group_name, feature_ids in groups.items()
    }


class PCVRDataConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    batch_size: int = 256
    num_workers: int = 16
    buffer_batches: int = 1
    train_steps_per_sweep: int = 0
    train_ratio: float = 1.0
    valid_ratio: float = 0.1
    split_strategy: PCVRDataSplitStrategy = "row_group_tail"
    sampling_strategy: PCVRDataSamplingStrategy = "step_random"
    eval_every_n_steps: int = 5_000
    seq_max_lens: str = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"

    @field_validator("split_strategy")
    @classmethod
    def _validate_split_strategy(cls, value: str) -> str:
        if value not in PCVR_DATA_SPLIT_STRATEGY_CHOICES:
            raise ValueError(f"unsupported PCVR data split strategy: {value}")
        return value

    @field_validator("sampling_strategy")
    @classmethod
    def _validate_sampling_strategy(cls, value: str) -> str:
        if value not in PCVR_DATA_SAMPLING_STRATEGY_CHOICES:
            raise ValueError(f"unsupported PCVR data sampling strategy: {value}")
        return value

    @field_validator("train_steps_per_sweep")
    @classmethod
    def _validate_train_steps_per_sweep(cls, value: int) -> int:
        if value < 0:
            raise ValueError("train_steps_per_sweep must be non-negative")
        return value

    @field_validator("eval_every_n_steps")
    @classmethod
    def _validate_eval_every_n_steps(cls, value: int) -> int:
        if value < 0:
            raise ValueError("eval_every_n_steps must be non-negative")
        return value


class PCVRSequenceCropConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["sequence_crop"] = "sequence_crop"
    enabled: bool = True
    views_per_row: int = 1
    seq_window_mode: PCVRSeqWindowMode = "tail"
    seq_window_min_len: int = 1


class PCVRFeatureMaskConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["feature_mask"] = "feature_mask"
    enabled: bool = True
    probability: float = 0.0


class PCVRDomainDropoutConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["domain_dropout"] = "domain_dropout"
    enabled: bool = True
    probability: float = 0.0


class PCVRNonSequentialSparseDropoutConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["nonseq_sparse_dropout"] = "nonseq_sparse_dropout"
    enabled: bool = True
    probability: float = 0.0


class PCVRDataCacheConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    mode: PCVRDataCacheMode = "none"
    max_batches: int = 0

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in PCVR_DATA_CACHE_MODE_CHOICES:
            raise ValueError(f"unsupported data cache mode: {value}")
        return value

    @property
    def enabled(self) -> bool:
        return self.mode != "none"


PCVRDataTransformConfig = (
    PCVRSequenceCropConfig
    | PCVRFeatureMaskConfig
    | PCVRDomainDropoutConfig
    | PCVRNonSequentialSparseDropoutConfig
)


class PCVRDataPipelineConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    cache: PCVRDataCacheConfig = Field(default_factory=PCVRDataCacheConfig)
    transforms: tuple[PCVRDataTransformConfig, ...] = ()
    seed: int | None = None
    strict_time_filter: bool = True

    @property
    def enabled(self) -> bool:
        return any(transform.enabled for transform in self.transforms)

    @property
    def transform_names(self) -> tuple[str, ...]:
        return tuple(
            transform.name for transform in self.transforms if transform.enabled
        )


class PCVROptimizerConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    lr: float = 1e-4
    max_steps: int = 0
    patience_steps: int = 25_000
    seed: int = 42
    device: str | None = None
    dense_optimizer_type: DenseOptimizerType = "adamw"
    scheduler_type: DenseLRSchedulerType = "none"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0

    @field_validator("dense_optimizer_type")
    @classmethod
    def _validate_dense_optimizer_type(cls, value: str) -> str:
        if value not in DENSE_OPTIMIZER_TYPE_CHOICES:
            raise ValueError(f"unsupported dense optimizer type: {value}")
        return value

    @field_validator("scheduler_type")
    @classmethod
    def _validate_scheduler_type(cls, value: str) -> str:
        if value not in DENSE_LR_SCHEDULER_TYPE_CHOICES:
            raise ValueError(f"unsupported scheduler type: {value}")
        return value

    @field_validator("patience_steps")
    @classmethod
    def _validate_patience_steps(cls, value: int) -> int:
        if value < 0:
            raise ValueError("patience_steps must be non-negative")
        return value

    @field_validator("warmup_steps")
    @classmethod
    def _validate_warmup_steps(cls, value: int) -> int:
        if value < 0:
            raise ValueError("warmup_steps must be non-negative")
        return value

    @field_validator("min_lr_ratio")
    @classmethod
    def _validate_min_lr_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("min_lr_ratio must be between 0.0 and 1.0")
        return value


class PCVREMAConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0
    update_every_n_steps: int = 1

    @field_validator("decay")
    @classmethod
    def _validate_decay(cls, value: float) -> float:
        if not 0.0 <= value < 1.0:
            raise ValueError("ema decay must be in [0.0, 1.0)")
        return value

    @field_validator("start_step")
    @classmethod
    def _validate_start_step(cls, value: int) -> int:
        if value < 0:
            raise ValueError("ema start_step must be non-negative")
        return value

    @field_validator("update_every_n_steps")
    @classmethod
    def _validate_update_every_n_steps(cls, value: int) -> int:
        if value < 1:
            raise ValueError("ema update_every_n_steps must be positive")
        return value


class PCVRSparseOptimizerConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    sparse_lr: float = 0.05
    sparse_weight_decay: float = 0.0
    reinit_sparse_every_n_steps: int = 0
    reinit_cardinality_threshold: int = 0


class PCVRNSConfig(TAACBoundaryModel):
    """Non-sequence token grouping configuration (model-side)."""

    model_config = ConfigDict(frozen=True)

    grouping_strategy: NSGroupingStrategy = "explicit"
    user_groups: dict[str, list[int]] = Field(default_factory=dict)
    item_groups: dict[str, list[int]] = Field(default_factory=dict)
    tokenizer_type: NSTokenizerType = "rankmixer"
    user_tokens: int = 0
    item_tokens: int = 0

    @field_validator("user_groups", "item_groups")
    @classmethod
    def _normalize_groups(cls, value: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
        return _normalize_ns_group_map(value)


class PCVRModelConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    d_model: int = 64
    emb_dim: int = 64
    num_queries: int = 1
    num_blocks: int = 2
    num_heads: int = 4
    seq_encoder_type: SeqEncoderType = "transformer"
    hidden_mult: int = 4
    dropout_rate: float = 0.01
    seq_top_k: int = 50
    seq_causal: bool = False
    action_num: int = 1
    use_time_buckets: bool = True
    rank_mixer_mode: RankMixerMode = "full"
    use_rope: bool = False
    rope_base: float = 10000.0
    emb_skip_threshold: int = 0
    seq_id_threshold: int = 10000
    gradient_checkpointing: bool = False
    flash_attention_backend: FlashAttentionBackend = "torch"
    rms_norm_backend: RMSNormBackend = "torch"
    rms_norm_block_rows: int = 1
    ns: PCVRNSConfig = Field(default_factory=PCVRNSConfig)

    @field_validator("flash_attention_backend")
    @classmethod
    def _validate_flash_attention_backend(cls, value: str) -> str:
        if value not in FLASH_ATTENTION_BACKEND_CHOICES:
            raise ValueError(f"unsupported flash attention backend: {value}")
        return value

    @field_validator("rms_norm_backend")
    @classmethod
    def _validate_rms_norm_backend(cls, value: str) -> str:
        if value not in RMS_NORM_BACKEND_CHOICES:
            raise ValueError(f"unsupported rms_norm backend: {value}")
        return value

    @field_validator("rms_norm_block_rows")
    @classmethod
    def _validate_rms_norm_block_rows(cls, value: int) -> int:
        if value < 1:
            raise ValueError("rms_norm_block_rows must be positive")
        return value


class PCVRValidationConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    early_stopping_metric: PCVREarlyStoppingMetric = "auc"

    @field_validator("early_stopping_metric")
    @classmethod
    def _validate_early_stopping_metric(cls, value: str) -> str:
        if value not in PCVR_EARLY_STOPPING_METRIC_CHOICES:
            raise ValueError(f"unsupported early stopping metric: {value}")
        return value


class PCVRTrainConfig(TAACBoundaryModel):
    """Full typed training configuration shared by all PCVR experiments."""

    model_config = ConfigDict(frozen=True)

    data: PCVRDataConfig = Field(default_factory=PCVRDataConfig)
    data_pipeline: PCVRDataPipelineConfig = Field(default_factory=PCVRDataPipelineConfig)
    optimizer: PCVROptimizerConfig = Field(default_factory=PCVROptimizerConfig)
    ema: PCVREMAConfig = Field(default_factory=PCVREMAConfig)
    runtime: RuntimeExecutionConfig = Field(default_factory=RuntimeExecutionConfig)
    loss: PCVRLossConfig = Field(default_factory=PCVRLossConfig)
    sparse_optimizer: PCVRSparseOptimizerConfig = Field(default_factory=PCVRSparseOptimizerConfig)
    model: PCVRModelConfig = Field(default_factory=PCVRModelConfig)
    validation: PCVRValidationConfig = Field(default_factory=PCVRValidationConfig)
