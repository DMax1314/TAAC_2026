from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    PCVRPredictionDataBundle,
    PCVRPredictionRunner,
    default_run_prediction_loop,
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
        request_timestamp=torch.tensor([100, 200][:batch_size], dtype=torch.long),
    )
    return PCVRBatch(
        inputs=inputs,
        label=torch.tensor(labels, dtype=torch.float32),
        user_id=list(user_ids),
    )


def test_default_prediction_loop_uses_lightweight_inference_payload(tmp_path: Path) -> None:
    observed_inference_mode: list[bool] = []
    batch = _make_batch(labels=[0.0, 1.0], user_ids=["u0", "u1"])

    def predict_fn(model_input: PCVRModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(model_input, PCVRModelInput)
        observed_inference_mode.append(torch.is_inference_mode_enabled())
        return torch.tensor([[-2.0], [2.0]]), torch.empty(2, 0)

    context = PCVRPredictionContext(
        model_module=SimpleNamespace(),
        model_class_name="DummyModel",
        model_type=torch.nn.Module,
        package_dir=tmp_path,
        dataset_path=tmp_path / "eval.parquet",
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
    data_bundle = PCVRPredictionDataBundle(dataset=SimpleNamespace(num_rows=2), loader=[batch])
    runner = PCVRPredictionRunner(model=object(), predict_fn=predict_fn)

    payload = default_run_prediction_loop(context, data_bundle, runner)

    assert observed_inference_mode == [True]
    assert payload["processed_rows"] == 2
    assert payload["batch_count"] == 1
    assert payload["predictions"] == {
        "u0": pytest.approx(0.11920292),
        "u1": pytest.approx(0.88079708),
    }
    assert "records" not in payload
    assert "labels" not in payload
    assert "probabilities" not in payload


def test_default_prediction_loop_keeps_evaluation_records(tmp_path: Path) -> None:
    batch = _make_batch(labels=[0.0, 1.0], user_ids=["u0", "u1"])

    def predict_fn(model_input: PCVRModelInput) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(model_input, PCVRModelInput)
        return torch.tensor([[-2.0], [2.0]]), torch.empty(2, 0)

    context = PCVRPredictionContext(
        model_module=SimpleNamespace(),
        model_class_name="DummyModel",
        model_type=torch.nn.Module,
        package_dir=tmp_path,
        dataset_path=tmp_path / "eval.parquet",
        schema_path=tmp_path / "schema.json",
        checkpoint_path=tmp_path / "checkpoint" / "model.safetensors",
        batch_size=2,
        num_workers=0,
        device="cpu",
        is_training_data=True,
        dataset_role="evaluation",
        config=PCVRTrainConfig(),
        runtime_execution=RuntimeExecutionConfig(compile=False),
    )
    data_bundle = PCVRPredictionDataBundle(dataset=SimpleNamespace(num_rows=2), loader=[batch])
    runner = PCVRPredictionRunner(model=object(), predict_fn=predict_fn)

    payload = default_run_prediction_loop(context, data_bundle, runner)

    assert [record["user_id"] for record in payload["records"]] == ["u0", "u1"]
    assert payload["labels"] == [0.0, 1.0]
    assert payload["probabilities"] == pytest.approx([0.11920292, 0.88079708])
    assert "predictions" not in payload
