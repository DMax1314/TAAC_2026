from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest
from taac2026.infrastructure.io.json import dumps, loads
from taac2026.domain.config import PCVRDataConfig, PCVRTrainConfig
from taac2026.domain.sidecar import build_pcvr_train_config_sidecar
import taac2026.application.experiments.experiment as experiment_module
from taac2026.application.experiments.experiment import PCVRExperiment, _log_prediction_progress
from taac2026.domain.runtime_config import RuntimeExecutionConfig


class DummyModel:
    num_ns = 0

    def __init__(self, *, schema, config):
        del schema, config

    def to(self, device):
        del device
        return self

    def parameters(self):
        return []


def _make_experiment(
    tmp_path: Path,
    *,
    train_defaults: PCVRTrainConfig | None = None,
) -> PCVRExperiment:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "model.py").write_text(
        "class DummyModel:\n    pass\n",
        encoding="utf-8",
    )
    return PCVRExperiment(
        name="pcvr_symbiosis",
        package_dir=package_dir,
        model_type=DummyModel,
        config_type=PCVRTrainConfig,
        train_defaults=train_defaults or PCVRTrainConfig(),
    )


def _write_observed_schema_fixture(schema_path: Path, parquet_path: Path) -> None:
    payload = {
        "format": "raw_parquet",
        "user_int": [[1, 10, 1], [2, 20, 4]],
        "item_int": [[3, 20, 1]],
        "user_dense": [[4, 4]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 0], [11, 20]],
            }
        },
    }
    schema_path.write_text(dumps(payload), encoding="utf-8")
    pq.write_table(
        pa.table(
            {
                "user_int_feats_1": [1, 2],
                "user_int_feats_2": [[1, 2], [2, 3, 4]],
                "item_int_feats_3": [10, 11],
                "user_dense_feats_4": [[0.1, 0.2], [0.3]],
                "domain_a_seq_10": [[100, 101], [103]],
                "domain_a_seq_11": [[5, 6], [6, 7, 7]],
                "timestamp": [100, 200],
            }
        ),
        parquet_path,
        row_group_size=1,
    )


def _write_train_config(checkpoint_dir: Path, overrides: dict[str, object] | None = None) -> None:
    config = PCVRTrainConfig()
    if overrides:
        payload = config.model_dump(mode="json")
        for section, values in overrides.items():
            payload.setdefault(section, {})
            if isinstance(values, dict):
                payload[section].update(values)
            else:
                payload[section] = values
        config = PCVRTrainConfig.model_validate(payload)
    (checkpoint_dir / "train_config.json").write_text(dumps(build_pcvr_train_config_sidecar(config)), encoding="utf-8")


def test_resolve_prediction_runtime_settings_requires_train_config_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=0, num_workers=8)),
    )
    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    with pytest.raises(ValueError, match="batch_size"):
        experiment._resolve_prediction_runtime_settings(request, experiment.train_defaults)


def test_resolve_prediction_runtime_settings_preserves_explicit_request_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=128, num_workers=8)),
    )
    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
        batch_size=64,
        num_workers=2,
    )

    resolved = experiment._resolve_prediction_runtime_settings(request, {"batch_size": 32, "num_workers": 6})

    assert resolved == (64, "request", 2, "request")


def test_infer_uses_train_config_runtime_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(data=PCVRDataConfig(batch_size=128, num_workers=8)),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    schema_payload = {"features": [{"name": "user_id"}]}
    (checkpoint_dir / "schema.json").write_text(dumps(schema_payload), encoding="utf-8")
    _write_train_config(checkpoint_dir, {"data": {"batch_size": 96, "num_workers": 4}})
    captured: dict[str, object] = {}

    def fake_run_prediction_loop(**kwargs):
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        return fake_run_prediction_loop(**kwargs)

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    payload = experiment.infer(request)

    assert captured["batch_size"] == 96
    assert captured["num_workers"] == 4
    assert payload["batch_size"] == 96
    assert payload["num_workers"] == 4
    assert payload["schema_path"] == str((checkpoint_dir / "schema.json").resolve())
    assert payload["schema"] == schema_payload
    assert payload["telemetry"]["label"] == "inference"
    assert payload["telemetry"]["rows"] == 1
    assert (tmp_path / "results" / "inference_telemetry.json").exists()
    assert loads((tmp_path / "results" / "predictions.json").read_bytes()) == {
        "predictions": {"u1": 0.5},
    }


def test_infer_consumes_lightweight_prediction_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    schema_payload = {"features": [{"name": "user_id"}]}
    (checkpoint_dir / "schema.json").write_text(dumps(schema_payload), encoding="utf-8")
    _write_train_config(checkpoint_dir, {"data": {"batch_size": 2, "num_workers": 0}})

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self, kwargs
        return {
            "predictions": {"u1": 0.25, "u2": 0.75},
            "processed_rows": 2,
            "batch_count": 1,
        }

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    payload = experiment.infer(
        InferRequest(
            experiment="experiments/symbiosis",
            dataset_path=tmp_path / "eval.parquet",
            schema_path=None,
            checkpoint_path=None,
            result_dir=tmp_path / "results",
        )
    )

    assert payload["prediction_count"] == 2
    assert payload["telemetry"]["rows"] == 2
    assert loads((tmp_path / "results" / "predictions.json").read_bytes()) == {
        "predictions": {"u1": 0.25, "u2": 0.75},
    }


def test_train_writes_split_observed_schema_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _make_experiment(tmp_path)
    dataset_path = tmp_path / "train.parquet"
    schema_path = tmp_path / "schema.json"
    run_dir = tmp_path / "outputs"
    _write_observed_schema_fixture(schema_path, dataset_path)

    monkeypatch.setattr(
        experiment_module,
        "train_pcvr_model",
        lambda **kwargs: {
            "run_dir": str(run_dir.resolve()),
            "checkpoint_root": str(run_dir.resolve()),
            "schema_path": str(schema_path.resolve()),
            "train_ratio": 1.0,
            "valid_ratio": 0.1,
        },
    )

    payload = experiment.train(
        TrainRequest(
            experiment="experiments/symbiosis",
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
        )
    )

    observed_paths = payload["observed_schema_paths"]
    assert set(observed_paths) == {"train_split", "valid_split"}
    train_report = loads(Path(observed_paths["train_split"]).read_bytes())
    valid_report = loads(Path(observed_paths["valid_split"]).read_bytes())
    assert train_report["dataset_role"] == "train_split"
    assert valid_report["dataset_role"] == "valid_split"
    assert payload["row_group_split"]["train_row_group_range"] == [0, 1]
    assert payload["row_group_split"]["valid_row_group_range"] == [1, 2]
    assert train_report["schema"]["user_int"] == [[1, 1, 1], [2, 2, 2]]
    assert valid_report["schema"]["user_int"] == [[1, 1, 1], [2, 3, 3]]


def test_train_writes_timestamp_auto_split_observed_schema_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _make_experiment(tmp_path)
    dataset_path = tmp_path / "train.parquet"
    schema_path = tmp_path / "schema.json"
    run_dir = tmp_path / "outputs"
    _write_observed_schema_fixture(schema_path, dataset_path)

    monkeypatch.setattr(
        experiment_module,
        "train_pcvr_model",
        lambda **kwargs: {
            "run_dir": str(run_dir.resolve()),
            "checkpoint_root": str(run_dir.resolve()),
            "schema_path": str(schema_path.resolve()),
            "train_ratio": 1.0,
            "valid_ratio": 0.1,
            "split_strategy": "timestamp_auto",
        },
    )

    payload = experiment.train(
        TrainRequest(
            experiment="experiments/symbiosis",
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
        )
    )

    observed_paths = payload["observed_schema_paths"]
    train_report = loads(Path(observed_paths["train_split"]).read_bytes())
    valid_report = loads(Path(observed_paths["valid_split"]).read_bytes())
    assert payload["data_split"] == {
        "split_strategy": "timestamp_auto",
        "train_timestamp_range": {"start": None, "end": 200},
        "valid_timestamp_range": {"start": 200, "end": None},
    }
    assert payload["row_group_split"]["train_row_group_range"] == [0, 2]
    assert payload["row_group_split"]["valid_row_group_range"] == [0, 2]
    assert payload["row_group_split"]["train_rows"] == 1
    assert payload["row_group_split"]["valid_rows"] == 1
    assert train_report["timestamp_range"] == {"start": None, "end": 200}
    assert valid_report["timestamp_range"] == {"start": 200, "end": None}
    assert train_report["schema"]["user_int"] == [[1, 1, 1], [2, 2, 2]]
    assert valid_report["schema"]["user_int"] == [[1, 1, 1], [2, 3, 3]]


def test_train_defaults_missing_dataset_to_hf_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    run_dir = tmp_path / "outputs"
    resolved_dataset_path = tmp_path / "hf_cache" / "demo_1000.parquet"
    resolved_schema_path = tmp_path / "hf_cache" / "schema.json"
    resolved_dataset_path.parent.mkdir(parents=True)
    _write_observed_schema_fixture(resolved_schema_path, resolved_dataset_path)
    captured_argv: dict[str, object] = {}

    def fake_train_pcvr_model(**kwargs):
        captured_argv["argv"] = kwargs["argv"]
        captured_argv["dataset_path"] = kwargs["dataset_path"]
        captured_argv["schema_path_override"] = kwargs["schema_path_override"]
        return {
            "run_dir": str(run_dir.resolve()),
            "checkpoint_root": str(run_dir.resolve()),
            "schema_path": str(resolved_schema_path.resolve()),
            "train_ratio": 1.0,
            "valid_ratio": 0.1,
        }

    monkeypatch.setattr(
        experiment_module,
        "resolve_default_pcvr_sample_paths",
        lambda dataset_path, schema_path: (resolved_dataset_path, resolved_schema_path),
    )
    monkeypatch.setattr(experiment_module, "train_pcvr_model", fake_train_pcvr_model)

    payload = experiment.train(
        TrainRequest(
            experiment="experiments/symbiosis",
            dataset_path=None,
            schema_path=None,
            run_dir=run_dir,
        )
    )

    assert captured_argv["dataset_path"] == resolved_dataset_path
    assert captured_argv["schema_path_override"] == resolved_schema_path
    assert set(payload["observed_schema_paths"]) == {"train_split", "valid_split"}


def test_infer_defaults_missing_dataset_to_hf_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    _write_train_config(checkpoint_dir, {"data": {"batch_size": 32, "num_workers": 1}})
    resolved_dataset_path = tmp_path / "hf_cache" / "demo_1000.parquet"
    resolved_dataset_path.parent.mkdir(parents=True)
    resolved_dataset_path.write_bytes(b"parquet")
    schema_payload = {"features": [{"name": "user_id"}]}
    resolved_schema_path = tmp_path / "hf_cache" / "schema.json"
    resolved_schema_path.write_text(dumps(schema_payload), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(
        experiment_module,
        "resolve_default_pcvr_sample_paths",
        lambda dataset_path, schema_path: (resolved_dataset_path, resolved_schema_path),
    )
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=None,
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    payload = experiment.infer(request)

    assert captured["dataset_path"] == resolved_dataset_path
    assert payload["schema_path"] == str(resolved_schema_path.resolve())


def test_resolve_prediction_runtime_execution_uses_train_config_values(tmp_path: Path) -> None:
    experiment = _make_experiment(
        tmp_path,
        train_defaults=PCVRTrainConfig(runtime=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)),
    )
    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    runtime_execution, amp_source, amp_dtype_source, compile_source = experiment._resolve_prediction_runtime_execution(
        request,
        experiment.train_defaults,
    )

    assert runtime_execution == RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)
    assert amp_source == "train_config"
    assert amp_dtype_source == "train_config"
    assert compile_source == "train_config"


def test_load_train_config_requires_sidecar(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match=r"train_config\.json"):
        experiment._load_train_config(checkpoint_dir)


def test_load_train_config_rejects_unknown_flat_keys(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    payload = build_pcvr_train_config_sidecar(PCVRTrainConfig())
    payload["train_config"]["amp"] = True
    (checkpoint_dir / "train_config.json").write_text(dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="amp"):
        experiment._load_train_config(checkpoint_dir)


def test_load_train_config_rejects_incomplete_sidecar(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    payload = build_pcvr_train_config_sidecar(PCVRTrainConfig())
    payload["train_config"].pop("data")
    (checkpoint_dir / "train_config.json").write_text(dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"incomplete train_config sidecar; missing fields:.*data"):
        experiment._load_train_config(checkpoint_dir)


def test_infer_uses_train_config_runtime_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")
    (checkpoint_dir / "schema.json").write_text(dumps({"features": []}), encoding="utf-8")
    _write_train_config(checkpoint_dir, {"runtime": {"amp": True, "amp_dtype": "float16", "compile": True}})
    captured: dict[str, object] = {}

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self
        captured.update(kwargs)
        return {"records": [{"user_id": "u1", "score": 0.5, "target": 0.0, "timestamp": None}]}

    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(checkpoint_dir))
    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)

    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
    )

    experiment.infer(request)

    runtime_execution = captured["runtime_execution"]
    assert runtime_execution == RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)


def test_evaluate_writes_score_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _make_experiment(tmp_path)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.safetensors"
    checkpoint_path.write_bytes(b"checkpoint")
    schema_payload = {
        "format": "raw_parquet",
        "user_int": [[1, 10, 1], [2, 20, 4]],
        "item_int": [[3, 20, 1]],
        "user_dense": [[4, 4]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 0], [11, 20]],
            }
        },
    }
    (checkpoint_dir / "schema.json").write_text(dumps(schema_payload), encoding="utf-8")
    _write_train_config(checkpoint_dir)
    dataset_path = tmp_path / "eval.parquet"
    _write_observed_schema_fixture(checkpoint_dir / "schema.json", dataset_path)

    def fake_bound_run_prediction_loop(self, **kwargs):
        del self, kwargs
        return {
            "labels": [0.0, 1.0, 1.0, 0.0],
            "probabilities": [0.1, 0.9, 0.8, 0.2],
            "records": [
                {"user_id": "u0", "score": 0.1, "target": 0.0, "timestamp": None},
                {"user_id": "u1", "score": 0.9, "target": 1.0, "timestamp": None},
            ],
        }

    monkeypatch.setattr(PCVRExperiment, "_run_prediction_loop", fake_bound_run_prediction_loop)
    output_path = tmp_path / "evaluation.json"
    request = EvalRequest(
        experiment="experiments/symbiosis",
        dataset_path=dataset_path,
        schema_path=None,
        run_dir=checkpoint_dir,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        predictions_path=tmp_path / "predictions.jsonl",
    )

    payload = experiment.evaluate(request)

    diagnostics = payload["metrics"]["score_diagnostics"]
    assert diagnostics["positive_count"] == 2
    assert diagnostics["negative_count"] == 2
    assert diagnostics["score_margin_mean"] == pytest.approx(0.7)
    assert payload["metrics"]["auc_ci"]["low"] <= payload["metrics"]["auc"] <= payload["metrics"]["auc_ci"]["high"]
    assert payload["telemetry"]["label"] == "evaluation"
    assert payload["telemetry"]["rows"] == 4
    assert payload["data_diagnostics"]["row_group_split"]["is_l1_ready"] is True
    assert payload["schema_path"] == str((checkpoint_dir / "schema.json").resolve())
    assert payload["schema"] == schema_payload
    observed_schema_path = Path(payload["observed_schema_paths"]["eval"])
    assert observed_schema_path.exists()
    observed_schema_payload = loads(observed_schema_path.read_bytes())
    assert observed_schema_payload["dataset_role"] == "eval"
    assert observed_schema_payload["schema"]["user_int"] == [[1, 2, 1], [2, 4, 3]]
    saved_payload = loads(output_path.read_bytes())
    assert saved_payload["metrics"]["score_diagnostics"] == diagnostics
    assert saved_payload["telemetry"]["label"] == "evaluation"
    assert saved_payload["data_diagnostics"] == payload["data_diagnostics"]
    assert saved_payload["schema"] == schema_payload
    assert saved_payload["observed_schema_paths"] == payload["observed_schema_paths"]
    predictions_payload = (tmp_path / "predictions.jsonl").read_bytes()
    assert predictions_payload.endswith(b"\n")
    assert [loads(line) for line in predictions_payload.splitlines()] == [
        {"user_id": "u0", "score": 0.1, "target": 0.0, "timestamp": None},
        {"user_id": "u1", "score": 0.9, "target": 1.0, "timestamp": None},
    ]
    assert (tmp_path / "evaluation_telemetry.json").exists()


def test_infer_request_runtime_settings_override_train_config(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path)
    request = InferRequest(
        experiment="experiments/symbiosis",
        dataset_path=tmp_path / "eval.parquet",
        schema_path=None,
        checkpoint_path=None,
        result_dir=tmp_path / "results",
        amp=False,
        amp_dtype="bfloat16",
        compile=False,
    )

    runtime_execution, amp_source, amp_dtype_source, compile_source = experiment._resolve_prediction_runtime_execution(
        request,
        PCVRTrainConfig(runtime=RuntimeExecutionConfig(amp=True, amp_dtype="float16", compile=True)),
    )

    assert runtime_execution == RuntimeExecutionConfig(amp=False, amp_dtype="bfloat16", compile=False)
    assert amp_source == "request"
    assert amp_dtype_source == "request"
    assert compile_source == "request"


def test_log_prediction_progress_reports_rows_and_batches(log_capture) -> None:
    with log_capture.at_level(logging.INFO):
        _log_prediction_progress(
            mode="inference",
            processed_rows=50_000,
            total_rows=310_000,
            batch_index=200,
            total_batches=1_211,
            elapsed_seconds=12.3,
        )

    assert "PCVR inference progress:" in log_capture.text
    assert "50000/310000 rows" in log_capture.text
    assert "batch 200/1211" in log_capture.text
    assert "elapsed=12.3s" in log_capture.text
