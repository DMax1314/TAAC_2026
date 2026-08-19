"""Stable public imports for experiment packages."""

from __future__ import annotations

from taac2026.application.experiments.experiment import create_pcvr_experiment
from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVREMAConfig,
    PCVRFeatureMaskConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVROptimizerConfig,
    PCVRSparseOptimizerConfig,
    PCVRSequenceCropConfig,
    PCVRTrainConfig,
)
from taac2026.infrastructure.data.batches import PCVRModelInput
from taac2026.infrastructure.modeling.model_contract import build_pcvr_model_specs
from taac2026.domain.runtime_config import PCVRLossConfig, PCVRLossTermConfig, RuntimeExecutionConfig
from taac2026.domain.experiment import FunctionExperiment
from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.schema import PCVRSchema
from taac2026.infrastructure.modeling import (
    DenseTokenProjector,
    EmbeddingParameterMixin,
    FeatureEmbeddingBank,
    LayerNorm,
    NonSequentialTokenizer,
    RMSNorm,
    SequenceTokenizer,
    causal_valid_attention_mask,
    choose_num_heads,
    deduplicate_sequence_events,
    make_padding_mask,
    masked_last,
    masked_mean,
    maybe_gradient_checkpoint,
    safe_key_padding_mask,
    scaled_dot_product_attention,
    sinusoidal_positions,
)
from taac2026.infrastructure.modeling.time_features import (
    NUM_TIME_BUCKETS,
    compute_sequence_stats,
    compute_sequence_time_buckets,
)


__all__ = [
    "NUM_TIME_BUCKETS",
    "DenseTokenProjector",
    "EmbeddingParameterMixin",
    "EvalRequest",
    "FeatureEmbeddingBank",
    "FunctionExperiment",
    "InferRequest",
    "LayerNorm",
    "NonSequentialTokenizer",
    "PCVRDataCacheConfig",
    "PCVRDataConfig",
    "PCVRDataPipelineConfig",
    "PCVRDomainDropoutConfig",
    "PCVREMAConfig",
    "PCVRFeatureMaskConfig",
    "PCVRLossConfig",
    "PCVRLossTermConfig",
    "PCVRModelConfig",
    "PCVRModelInput",
    "PCVRNSConfig",
    "PCVRNonSequentialSparseDropoutConfig",
    "PCVROptimizerConfig",
    "PCVRSchema",
    "PCVRSequenceCropConfig",
    "PCVRSparseOptimizerConfig",
    "PCVRTrainConfig",
    "RMSNorm",
    "RuntimeExecutionConfig",
    "SequenceTokenizer",
    "TrainRequest",
    "build_pcvr_model_specs",
    "causal_valid_attention_mask",
    "choose_num_heads",
    "compute_sequence_stats",
    "compute_sequence_time_buckets",
    "create_pcvr_experiment",
    "deduplicate_sequence_events",
    "make_padding_mask",
    "masked_last",
    "masked_mean",
    "maybe_gradient_checkpoint",
    "safe_key_padding_mask",
    "scaled_dot_product_attention",
    "sinusoidal_positions",
]
