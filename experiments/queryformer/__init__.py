"""Clean-room QueryFormer experiment from the TAAC 2026 champion write-up.

The public article describes the architecture and ablations but does not
publish source code. This package therefore implements those documented
contracts on the shared PCVR runtime instead of claiming to port official
code.
"""

from __future__ import annotations

from pathlib import Path

from taac2026.api import (
    PCVRDataCacheConfig,
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVREMAConfig,
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRNSConfig,
    PCVROptimizerConfig,
    PCVRSparseOptimizerConfig,
    RuntimeExecutionConfig,
    create_pcvr_experiment,
)

from .config import QueryFormerModelConfig, QueryFormerTrainConfig
from .model import PCVRQueryFormer

TRAIN_DEFAULTS = QueryFormerTrainConfig(
    data=PCVRDataConfig(
        batch_size=256,
        num_workers=8,
        buffer_batches=8,
        train_ratio=1.0,
        valid_ratio=0.1,
        split_strategy="timestamp_auto",
        sampling_strategy="step_random",
        eval_every_n_steps=5_000,
        seq_max_lens="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
    ),
    data_pipeline=PCVRDataPipelineConfig(
        cache=PCVRDataCacheConfig(mode="none", max_batches=0),
        transforms=(),
        seed=2026,
        strict_time_filter=True,
    ),
    optimizer=PCVROptimizerConfig(
        lr=1e-4,
        max_steps=100_000,
        patience_steps=25_000,
        seed=2026,
        device=None,
        dense_optimizer_type="muon",
        scheduler_type="cosine",
        warmup_steps=0,
        min_lr_ratio=0.0,
    ),
    ema=PCVREMAConfig(enabled=True, decay=0.999, start_step=0),
    runtime=RuntimeExecutionConfig(
        amp=True,
        amp_dtype="bfloat16",
        compile=True,
        progress_log_interval_steps=100,
    ),
    loss=PCVRLossConfig(
        terms=(PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),)
    ),
    sparse_optimizer=PCVRSparseOptimizerConfig(
        sparse_lr=0.05,
        sparse_weight_decay=0.0,
        reinit_sparse_every_n_steps=0,
        reinit_cardinality_threshold=0,
    ),
    model=QueryFormerModelConfig(
        d_model=128,
        emb_dim=64,
        num_queries=1,
        num_blocks=2,
        num_heads=4,
        hidden_mult=4,
        dropout_rate=0.02,
        action_num=1,
        use_time_buckets=True,
        emb_skip_threshold=1_000_000,
        gradient_checkpointing=False,
        num_embedding_columns=4,
        dcn_num_layers=2,
        compress_high_cardinality=True,
        use_query_self_attention=True,
        use_query_cross_attention=True,
        use_query_seq_cross_attention=True,
        use_seq_query_cross_attention=True,
        ns=PCVRNSConfig(
            # QueryFormer uses the official grouped sparse-token contract;
            # dense fields keep their original schema boundaries in the model.
            grouping_strategy="explicit",
            user_groups={
                "U1": [1, 15],
                "U2": [48, 49, 89, 90, 91],
                "U3": [80],
                "U4": [51, 52, 53, 54, 86],
                "U5": [82, 92, 93],
                "U6": [
                    50,
                    60,
                    94,
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                ],
                "U7": [3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66],
            },
            item_groups={
                "I1": [11, 13],
                "I2": [5, 6, 7, 8, 12],
                "I3": [16, 81, 83, 84, 85],
                "I4": [9, 10],
            },
            tokenizer_type="group",
            user_tokens=0,
            item_tokens=0,
        ),
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_queryformer",
    package_dir=Path(__file__).resolve().parent,
    model_type=PCVRQueryFormer,
    config_type=QueryFormerTrainConfig,
    train_defaults=TRAIN_DEFAULTS,
)

__all__ = ["EXPERIMENT", "TRAIN_DEFAULTS"]
