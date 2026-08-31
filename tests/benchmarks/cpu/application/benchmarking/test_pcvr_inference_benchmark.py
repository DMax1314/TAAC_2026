from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from taac2026.application.benchmarking.pcvr_inference_benchmark import (
    _latency_stats,
    _measured_inference_pass,
    parse_args,
)
from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    PCVRPredictionDataBundle,
    PCVRPredictionRunner,
)
from taac2026.domain.config import PCVRTrainConfig
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVREntityInput,
    PCVRModelInput,
)


def _make_batch(labels: list[float], user_ids: list[str]) -> PCVRBatch:
    batch_size = len(labels)
    empty = PCVREntityInput(
        int_values=torch.ones(batch_size, 1, dtype=torch.long),
        int_missing_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
        dense_values=torch.zeros(batch_size, 1),
        dense_missing_mask=torch.zeros(batch_size, 1, dtype=torch.bool),
    )
    inputs = PCVRModelInput(
        user=empty,
        item=empty,
        sequences={},
        request_timestamp=torch.arange(batch_size, dtype=torch.long),
    )
    return PCVRBatch(
        inputs=inputs,
        label=torch.tensor(labels, dtype=torch.float32),
        user_id=list(user_ids),
    )


def _make_context(tmp_path: Path) -> PCVRPredictionContext:
    return PCVRPredictionContext(
        model_type=torch.nn.Module,
        dataset_path=tmp_path / "infer.parquet",
        schema_path=tmp_path / "schema.json",
        checkpoint_path=tmp_path / "checkpoint" / "model.safetensors",
        batch_size=2,
        num_workers=0,
        device="cpu",
        is_training_data=False,
        dataset_role="inference",
        config=PCVRTrainConfig(),
        runtime_execution=RuntimeExecutionConfig(compile=False),
    )


def _make_runner() -> PCVRPredictionRunner:
    def predict_fn(model_input: PCVRModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        rows = model_input.request_timestamp.shape[0]
        return torch.zeros(rows, 1), torch.empty(rows, 0)

    return PCVRPredictionRunner(model=object(), predict_fn=predict_fn)


def test_latency_stats_empty() -> None:
    assert _latency_stats([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}


def test_latency_stats_values() -> None:
    stats = _latency_stats([10.0, 20.0, 30.0])
    assert stats["mean"] == 20.0
    assert stats["p50"] == 20.0
    assert stats["max"] == 30.0


def test_measured_inference_pass_splits_warmup_and_steady(tmp_path: Path) -> None:
    batches = [_make_batch([0.0, 1.0], ["u0", "u1"]) for _ in range(4)]
    data_bundle = PCVRPredictionDataBundle(dataset=SimpleNamespace(num_rows=8), loader=batches)

    result = _measured_inference_pass(
        _make_context(tmp_path),
        data_bundle,
        _make_runner(),
        warmup_batches=1,
        max_batches=0,
    )

    assert result["rows"] == 8
    assert result["batches"] == 4
    assert result["warmup_rows"] == 2
    assert result["steady_rows"] == 6
    assert result["elapsed_sec"] > 0.0
    assert result["rows_per_sec"] > 0.0
    assert result["steady_rows_per_sec"] > 0.0
    assert result["batch_latency_ms"]["mean"] >= 0.0


def test_measured_inference_pass_respects_max_batches(tmp_path: Path) -> None:
    batches = [_make_batch([0.0, 1.0], ["u0", "u1"]) for _ in range(10)]
    data_bundle = PCVRPredictionDataBundle(dataset=SimpleNamespace(num_rows=20), loader=batches)

    result = _measured_inference_pass(
        _make_context(tmp_path),
        data_bundle,
        _make_runner(),
        warmup_batches=2,
        max_batches=3,
    )

    assert result["batches"] == 5
    assert result["rows"] == 10
    assert result["warmup_rows"] == 4
    assert result["steady_rows"] == 6


def test_parse_args_defaults() -> None:
    args = parse_args([
        "--experiment",
        "experiments/baseline",
        "--checkpoint",
        "outputs/bench_infer_ckpt",
    ])

    assert args.experiment == "experiments/baseline"
    assert args.batch_size == 256
    assert args.num_workers == 0
    assert args.amp is False
    assert args.amp_dtype == "bfloat16"
    assert args.compile is False
    assert args.warmup_batches == 3
    assert args.max_batches == 0
    assert args.repeats == 1
    assert args.json is False
