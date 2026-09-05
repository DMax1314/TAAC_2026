"""Symbiosis PCVR experiment package."""

from __future__ import annotations

from pathlib import Path

from taac2026.api import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRNSConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVROptimizerConfig,
    PCVRSparseOptimizerConfig,
)
from taac2026.api import create_pcvr_experiment
from taac2026.api import RuntimeExecutionConfig
from .config import SymbiosisModelConfig, SymbiosisTrainConfig
from .model import PCVRSymbiosis

__all__ = [
    "EXPERIMENT",
    "TRAIN_DEFAULTS",
    "SymbiosisModelConfig",
    "SymbiosisTrainConfig",
]


TRAIN_DEFAULTS = SymbiosisTrainConfig(
    data=PCVRDataConfig(
        batch_size=128,
        num_workers=8,
        buffer_batches=1,
        train_ratio=1.0,
        valid_ratio=0.1,
        split_strategy="timestamp_auto",
        sampling_strategy="row_group_sweep",
        eval_every_n_steps=500,
        seq_max_lens="seq_a:256,seq_b:256,seq_c:1024,seq_d:2048",
    ),
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="none", max_batches=0),
        transforms=(
            PCVRFeatureMaskConfig(probability=0.03),
            PCVRNonSequentialSparseDropoutConfig(probability=0.10),
            PCVRDomainDropoutConfig(probability=0.02),
        ),
        seed=42,
        strict_time_filter=True,
    ),
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=100_000,
        patience_steps=2500,
        seed=42,
        device=None,
        dense_optimizer_type="muon",
        scheduler_type="none",
        warmup_steps=100,
        min_lr_ratio=0.1,
    ),
    runtime=RuntimeExecutionConfig(
        amp=True,
        amp_dtype="bfloat16",
        compile=True,
        progress_log_interval_steps=100,
    ),
    loss=PCVRLossConfig(terms=(PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),)),
    sparse_optimizer=PCVRSparseOptimizerConfig(
        sparse_lr=0.01,
        reinit_sparse_every_n_steps=0,
        reinit_cardinality_threshold=0,
    ),
    model=SymbiosisModelConfig(
        d_model=256,
        emb_dim=256,
        num_queries=1,
        num_blocks=4,
        num_heads=8,
        seq_encoder_type="transformer",
        hidden_mult=4,
        dropout_rate=0.04,
        seq_top_k=64,
        seq_causal=False,
        action_num=1,
        use_time_buckets=True,
        rank_mixer_mode="full",
        use_rope=False,
        rope_base=10000.0,
        emb_skip_threshold=1_000_000,
        seq_id_threshold=10000,
        gradient_checkpointing=False,
        ns=PCVRNSConfig(
            # NS token groups for parquet data. Values are fids, using the numeric suffix
            # from user_int_feats_{fid} or item_int_feats_{fid}; runtime converts them to
            # schema entry indices at load time. For the official schema, num_ns is
            # 7 user-int groups + 1 user-dense token + 4 item-int groups = 12 before
            # model-specific tokenizer changes. User dense features are projected to one
            # dense NS token and are not configured in this mapping. Fids
            # 62/63/64/65/66/89/90/91 also appear in user_dense_feats; the int parts are
            # grouped here, while the float parts stay in the single user_dense NS token.
            grouping_strategy="explicit",
            user_groups={
                "U1": [1, 15],
                "U2": [48, 49, 89, 90, 91],
                "U3": [80],
                "U4": [51, 52, 53, 54, 86],
                "U5": [82, 92, 93],
                "U6": [50, 60, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "U7": [3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66],
            },
            item_groups={
                "I1": [11, 13],
                "I2": [5, 6, 7, 8, 12],
                "I3": [16, 81, 83, 84, 85],
                "I4": [9, 10],
            },
            tokenizer_type="rankmixer",
            user_tokens=7,
            item_tokens=4,
        ),
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_symbiosis",
    package_dir=Path(__file__).resolve().parent,
    model_type=PCVRSymbiosis,
    train_defaults=TRAIN_DEFAULTS,
    config_type=SymbiosisTrainConfig,
)
