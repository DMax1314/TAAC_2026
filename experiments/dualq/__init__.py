"""DualQ PCVR experiment package.

Ports https://github.com/zzhlkw-ai/TAAC2026 (MIT License) ``round2best``
(ported as ``dualq``; TAAC2026 academic track Round-2 Top-17, Unified
Module Innovation Award) onto the shared PCVR runtime. The model class
lives in ``model.py``; this module wires training defaults and the
experiment declaration only. Pair feature diversion and all time features
are resolved inside the model from ``PCVRSchema`` and ``PCVRModelInput``.
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
)
from taac2026.api import RuntimeExecutionConfig, create_pcvr_experiment

from .config import DualQModelConfig, DualQTrainConfig
from .model import PCVRDualQ

TRAIN_DEFAULTS = DualQTrainConfig(
    data=PCVRDataConfig(
        batch_size=576,
        num_workers=6,
        buffer_batches=32,
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
        seed=7789,
        strict_time_filter=True,
    ),
    optimizer=PCVROptimizerConfig(
        lr=4.83e-4,
        max_steps=100_000,
        patience_steps=25_000,
        seed=7789,
        device=None,
        dense_optimizer_type="adamw",
        scheduler_type="none",
        warmup_steps=0,
        min_lr_ratio=0.0,
    ),
    ema=PCVREMAConfig(
        enabled=True,
        decay=0.999,
        start_step=1500,
    ),
    runtime=RuntimeExecutionConfig(
        amp=True,
        amp_dtype="bfloat16",
        compile=True,
        progress_log_interval_steps=100,
    ),
    loss=PCVRLossConfig(
        terms=(PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),),
    ),
    sparse_optimizer=PCVRSparseOptimizerConfig(
        sparse_lr=0.05,
        sparse_weight_decay=0.0,
        reinit_sparse_every_n_steps=0,
        reinit_cardinality_threshold=0,
    ),
    model=DualQModelConfig(
        d_model=192,
        emb_dim=64,
        num_queries=6,
        num_blocks=2,
        num_heads=4,
        seq_encoder_type="swiglu",
        hidden_mult=4,
        dropout_rate=0.02,
        seq_top_k=50,
        seq_causal=False,
        action_num=1,
        use_time_buckets=True,
        rank_mixer_mode="full",
        use_rope=False,
        rope_base=10000.0,
        emb_skip_threshold=1_000_000,
        seq_id_threshold=10000,
        gradient_checkpointing=False,
        # dualq switches (source run.sh flags).
        use_global_time_token=True,
        use_seq_gap_buckets=True,
        use_time_gap_domain_gates=True,
        use_fid87_token_residual=True,
        use_time_decay_summary=True,
        user_q_tokens=4,
        item_q_tokens=2,
        use_time_aligned_interleave=True,
        seq_interest_ratios="1.0,0.7",
        pair_feature_fids="62,63,64,65,66,89,90,91",
        ns=PCVRNSConfig(
            # dualq uses singleton NS groups, same as round1best. Pair
            # fids 62-66/89-91 are diverted by the model and embedded by the
            # cross pair tokenizer instead.
            grouping_strategy="explicit",
            user_groups={
                "U1": [1],
                "U3": [3],
                "U4": [4],
                "U15": [15],
                "U48": [48],
                "U49": [49],
                "U50": [50],
                "U51": [51],
                "U52": [52],
                "U53": [53],
                "U54": [54],
                "U55": [55],
                "U56": [56],
                "U57": [57],
                "U58": [58],
                "U59": [59],
                "U60": [60],
                "U80": [80],
                "U82": [82],
                "U86": [86],
                "U92": [92],
                "U93": [93],
                "U94": [94],
                "U95": [95],
                "U96": [96],
                "U97": [97],
                "U98": [98],
                "U99": [99],
                "U100": [100],
                "U101": [101],
                "U102": [102],
                "U103": [103],
                "U104": [104],
                "U105": [105],
                "U106": [106],
                "U107": [107],
                "U108": [108],
                "U109": [109],
            },
            item_groups={
                "I5": [5],
                "I6": [6],
                "I7": [7],
                "I8": [8],
                "I9": [9],
                "I10": [10],
                "I11": [11],
                "I12": [12],
                "I13": [13],
                "I16": [16],
                "I81": [81],
                "I83": [83],
                "I84": [84],
                "I85": [85],
            },
            tokenizer_type="rankmixer",
            user_tokens=14,
            item_tokens=3,
        ),
    ),
)

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_dualq",
    package_dir=Path(__file__).resolve().parent,
    model_type=PCVRDualQ,
    config_type=DualQTrainConfig,
    train_defaults=TRAIN_DEFAULTS,
)

__all__ = ["EXPERIMENT", "TRAIN_DEFAULTS"]
