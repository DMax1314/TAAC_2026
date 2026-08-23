from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from taac2026.application.experiments.experiment import create_pcvr_experiment
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.domain.config import (
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRModelConfig,
    PCVRNSConfig,
    PCVROptimizerConfig,
    PCVRTrainConfig,
)
from taac2026.domain.requests import InferRequest, TrainRequest
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.checkpoints import PRIMARY_CHECKPOINT_FILENAME
from taac2026.infrastructure.io.json import dumps, loads


def _write_tiny_pcvr_dataset(root: Path) -> tuple[Path, Path]:
    dataset_path = root / "tiny.parquet"
    schema_path = root / "schema.json"
    schema_path.write_text(
        dumps(
            {
                "format": "raw_parquet",
                "user_int": [[1, 8, 1], [2, 8, 2]],
                "item_int": [[3, 8, 1]],
                "user_dense": [[4, 2]],
                "item_dense": [[5, 1]],
                "seq": {
                    "seq_a": {
                        "prefix": "domain_a_seq",
                        "ts_fid": 10,
                        "features": [[10, 0], [11, 8]],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "timestamp": list(range(100, 108)),
                "label_type": [2, 0, 2, 0, 2, 0, 2, 0],
                "user_id": [f"user-{index}" for index in range(8)],
                "user_int_feats_1": [1, 2, 3, 4, 5, 6, 7, 1],
                "user_int_feats_2": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 1], [1, 3]],
                "item_int_feats_3": [1, 2, 3, 4, 5, 6, 7, 1],
                "user_dense_feats_4": [[index / 10, (index + 1) / 10] for index in range(8)],
                "item_dense_feats_5": [[index / 20] for index in range(8)],
                "domain_a_seq_10": [[90 + index, 95 + index] for index in range(8)],
                "domain_a_seq_11": [[1 + index % 6, 2 + index % 6] for index in range(8)],
            }
        ),
        dataset_path,
        row_group_size=4,
    )
    return dataset_path, schema_path


def _tiny_baseline_experiment():
    baseline = load_experiment_package("experiments/baseline")
    defaults = PCVRTrainConfig(
        data=PCVRDataConfig(
            batch_size=2,
            num_workers=0,
            train_ratio=1.0,
            valid_ratio=0.5,
            split_strategy="row_group_tail",
            sampling_strategy="row_group_sweep",
            eval_every_n_steps=1,
            seq_max_lens="seq_a:4",
        ),
        data_pipeline=PCVRDataPipelineConfig(strict_time_filter=True),
        optimizer=PCVROptimizerConfig(
            lr=1e-3,
            max_steps=1,
            patience_steps=2,
            seed=7,
            device="cpu",
        ),
        runtime=RuntimeExecutionConfig(
            amp=False,
            compile=False,
            progress_log_interval_steps=1,
        ),
        model=PCVRModelConfig(
            d_model=16,
            emb_dim=8,
            num_queries=1,
            num_blocks=1,
            num_heads=2,
            seq_encoder_type="swiglu",
            hidden_mult=2,
            dropout_rate=0.0,
            seq_top_k=4,
            action_num=1,
            use_time_buckets=False,
            rank_mixer_mode="none",
            gradient_checkpointing=False,
            ns=PCVRNSConfig(
                grouping_strategy="singleton",
                tokenizer_type="rankmixer",
                user_tokens=2,
                item_tokens=1,
            ),
        ),
    )
    return create_pcvr_experiment(
        name="pcvr_baseline_roundtrip",
        package_dir=baseline.package_dir,
        model_type=baseline.model_type,
        train_defaults=defaults,
    )


def test_real_pcvr_cpu_training_checkpoint_and_inference_round_trip(tmp_path: Path) -> None:
    dataset_path, schema_path = _write_tiny_pcvr_dataset(tmp_path)
    run_dir = tmp_path / "run"
    result_dir = tmp_path / "inference"
    experiment = _tiny_baseline_experiment()

    train_summary = experiment.train(
        TrainRequest(
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
        )
    )

    checkpoints = list(run_dir.glob(f"global_step*/{PRIMARY_CHECKPOINT_FILENAME}"))
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert (checkpoint.parent / "schema.json").is_file()
    assert (checkpoint.parent / "train_config.json").is_file()
    assert (run_dir / "training_summary.json").is_file()
    assert train_summary["row_group_split"]["is_disjoint"] is True

    inference = experiment.infer(
        InferRequest(
            dataset_path=dataset_path,
            schema_path=None,
            checkpoint_path=checkpoint,
            result_dir=result_dir,
            batch_size=2,
            num_workers=0,
            device="cpu",
            amp=False,
            compile=False,
        )
    )

    predictions = loads((result_dir / "predictions.json").read_bytes())["predictions"]
    assert inference["prediction_count"] == 8
    assert set(predictions) == {f"user-{index}" for index in range(8)}
    assert all(0.0 <= score <= 1.0 for score in predictions.values())
