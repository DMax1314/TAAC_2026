"""PCVR experiment adapter for plugin packages."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.metrics import compute_classification_metrics
from taac2026.domain.runtime_config import RuntimeExecutionConfig, normalize_amp_dtype
from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.io.json import dump_bytes
from taac2026.domain.config import PCVRTrainConfig
from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    _log_prediction_progress,
    build_prediction_data,
    build_prediction_model,
    prepare_prediction_runner,
    run_prediction_loop,
)
from taac2026.application.evaluation.runtime import (
    build_evaluation_data_diagnostics,
    load_train_config,
    load_runtime_schema,
    resolve_evaluation_checkpoint,
    resolve_inference_checkpoint,
    write_observed_schema_report,
    write_train_split_observed_schema_reports,
)
from taac2026.infrastructure.data.sample_dataset import resolve_default_pcvr_sample_paths
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.runtime.telemetry import RuntimeTelemetry, file_size_mb
from taac2026.application.training.args import train_pcvr_model


_EVAL_AUC_BOOTSTRAP_SAMPLES = 200
_INFER_REQUEST_DEFAULT_BATCH_SIZE = int(InferRequest.__dataclass_fields__["batch_size"].default)
_INFER_REQUEST_DEFAULT_NUM_WORKERS = int(InferRequest.__dataclass_fields__["num_workers"].default)


def _callable_name(value: Any) -> str:
    return getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))


def create_pcvr_experiment(
    *,
    name: str,
    package_dir: Path,
    model_type: type[torch.nn.Module],
    train_defaults: PCVRTrainConfig,
    config_type: type[PCVRTrainConfig] = PCVRTrainConfig,
) -> PCVRExperiment:
    return PCVRExperiment(
        name=name,
        package_dir=package_dir,
        model_type=model_type,
        config_type=config_type,
        train_defaults=train_defaults,
    )


@dataclass(slots=True)
class _PredictionRun:
    checkpoint: Path
    dataset_path: Path
    schema_path: Path
    schema: Any
    batch_size: int
    num_workers: int
    telemetry: RuntimeTelemetry
    result: dict[str, Any]


@dataclass(slots=True)
class PCVRExperiment:
    kind = "pcvr"
    requires_dataset = True

    name: str
    package_dir: Path
    model_type: type[torch.nn.Module]
    config_type: type[PCVRTrainConfig]
    train_defaults: PCVRTrainConfig

    @property
    def model_class_name(self) -> str:
        return self.model_type.__name__

    @property
    def config_type_name(self) -> str:
        return _callable_name(self.config_type)

    def _configured_infer_runtime_value(
        self,
        config: PCVRTrainConfig,
        *,
        config_key: str,
        minimum: int,
    ) -> tuple[int, str]:
        configured_value = getattr(config.data, config_key)
        if configured_value is None or configured_value < minimum:
            raise ValueError(f"PCVR train_config key {config_key!r} must be >= {minimum}, got {configured_value!r}")
        return configured_value, "train_config"

    def _resolve_prediction_runtime_settings(
        self,
        request: EvalRequest | InferRequest,
        config: PCVRTrainConfig,
    ) -> tuple[int, str, int, str]:
        batch_size = int(request.batch_size)
        batch_size_source = "request" if request.batch_size != _INFER_REQUEST_DEFAULT_BATCH_SIZE else "cli_default"
        if batch_size_source == "cli_default":
            batch_size, batch_size_source = self._configured_infer_runtime_value(
                config,
                config_key="batch_size",
                minimum=1,
            )

        num_workers = int(request.num_workers)
        num_workers_source = "request" if request.num_workers != _INFER_REQUEST_DEFAULT_NUM_WORKERS else "cli_default"
        if num_workers_source == "cli_default":
            num_workers, num_workers_source = self._configured_infer_runtime_value(
                config,
                config_key="num_workers",
                minimum=0,
            )

        return batch_size, batch_size_source, num_workers, num_workers_source

    def _resolve_prediction_runtime_execution(
        self,
        request: EvalRequest | InferRequest,
        config: PCVRTrainConfig,
    ) -> tuple[RuntimeExecutionConfig, str, str, str]:
        runtime = config.runtime
        amp = runtime.amp if request.amp is None else request.amp
        amp_source = "train_config" if request.amp is None else "request"
        amp_dtype = runtime.amp_dtype if request.amp_dtype in (None, "") else normalize_amp_dtype(request.amp_dtype)
        amp_dtype_source = "train_config" if request.amp_dtype in (None, "") else "request"
        compile_enabled = runtime.compile if request.compile is None else request.compile
        compile_source = "train_config" if request.compile is None else "request"
        return (
            RuntimeExecutionConfig(
                amp=amp,
                amp_dtype=amp_dtype,
                compile=compile_enabled,
                deterministic=runtime.deterministic,
            ),
            amp_source,
            amp_dtype_source,
            compile_source,
        )

    def _run_prediction_loop(
        self,
        *,
        dataset_path: Path,
        schema_path: Path | None,
        checkpoint_path: Path,
        batch_size: int,
        num_workers: int,
        device: str,
        is_training_data: bool,
        dataset_role: str,
        config: PCVRTrainConfig | None = None,
        runtime_execution: RuntimeExecutionConfig | None = None,
    ) -> dict[str, Any]:
        context = PCVRPredictionContext(
            model_type=self.model_type,
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            is_training_data=is_training_data,
            dataset_role=dataset_role,
            config=config if config is not None else load_train_config(self.config_type, checkpoint_path.parent),
            runtime_execution=runtime_execution or RuntimeExecutionConfig(),
        )
        data_bundle = build_prediction_data(context)
        model = build_prediction_model(context, data_bundle)
        runner = prepare_prediction_runner(context, data_bundle, model)
        return run_prediction_loop(context, data_bundle, runner)

    def train(self, request: TrainRequest) -> Mapping[str, Any]:
        resolved_dataset_path, resolved_schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        run_dir = request.run_dir.expanduser().resolve()
        train_log_dir = Path(os.environ.get("TRAIN_LOG_PATH", str(run_dir / "logs"))).expanduser().resolve()
        tensorboard_dir = Path(os.environ.get("TRAIN_TF_EVENTS_PATH", str(run_dir / "tensorboard"))).expanduser().resolve()
        dataset_path = Path(os.environ.get("TRAIN_DATA_PATH", str(resolved_dataset_path))).expanduser().resolve()
        ckpt_dir = Path(os.environ.get("TRAIN_CKPT_PATH", str(run_dir))).expanduser().resolve()
        schema_override = resolved_schema_override
        env_schema_path = os.environ.get("TAAC_SCHEMA_PATH")
        if env_schema_path:
            schema_override = Path(env_schema_path)

        summary = dict(train_pcvr_model(
            model_class_name=self.model_class_name,
            model_type=self.model_type,
            defaults=self.train_defaults,
            config_type=self.config_type,
            argv=request.extra_args,
            dataset_path=dataset_path,
            schema_path_override=schema_override,
            ckpt_dir=ckpt_dir,
            log_dir=train_log_dir,
            tf_events_dir=tensorboard_dir,
        ) or {})

        resolved_schema_path = Path(summary["schema_path"]).expanduser().resolve()

        observed_schema_payload = write_train_split_observed_schema_reports(
            dataset_path=dataset_path,
            schema_path=resolved_schema_path,
            run_dir=run_dir,
            valid_ratio=float(summary["valid_ratio"]),
            train_ratio=float(summary["train_ratio"]),
            split_strategy=str(summary.get("split_strategy", "row_group_tail")),
            split_seed=int(summary.get("split_seed", 42)),
        )

        payload = dict(summary)
        payload["experiment_name"] = self.name
        payload["run_dir"] = str(run_dir)
        payload["checkpoint_root"] = str(ckpt_dir)
        payload["schema_path"] = str(resolved_schema_path)
        payload.update(observed_schema_payload)
        return payload

    def _execute_prediction_run(
        self,
        request: EvalRequest | InferRequest,
        *,
        checkpoint: Path,
        mode: str,
        is_training_data: bool,
    ) -> _PredictionRun:
        dataset_path, schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        config = load_train_config(self.config_type, checkpoint.parent)
        schema_path, schema = load_runtime_schema(
            schema_path=schema_override,
            checkpoint_dir=checkpoint.parent,
            mode=mode,
        )
        batch_size, batch_size_source, num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logger.info(
            "Resolved PCVR " + mode + " runtime: experiment={}, checkpoint={}, batch_size={} ({}), num_workers={} ({}), amp={} ({}), amp_dtype={} ({}), compile={} ({})",
            self.name,
            checkpoint,
            batch_size,
            batch_size_source,
            num_workers,
            num_workers_source,
            runtime_execution.amp,
            amp_source,
            runtime_execution.normalized_amp_dtype(),
            amp_dtype_source,
            runtime_execution.compile,
            compile_source,
        )

        telemetry = RuntimeTelemetry(
            label=mode,
            device=request.device,
            metadata={
                "experiment_name": self.name,
                "checkpoint_path": str(checkpoint),
                "dataset_path": str(dataset_path),
            },
        ).start()
        result = self._run_prediction_loop(
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_path=checkpoint,
            batch_size=batch_size,
            num_workers=num_workers,
            device=request.device,
            is_training_data=is_training_data,
            dataset_role=mode,
            config=config,
            runtime_execution=runtime_execution,
        )
        return _PredictionRun(
            checkpoint=checkpoint,
            dataset_path=dataset_path,
            schema_path=schema_path,
            schema=schema,
            batch_size=batch_size,
            num_workers=num_workers,
            telemetry=telemetry,
            result=result,
        )

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]:
        run = self._execute_prediction_run(
            request,
            checkpoint=resolve_evaluation_checkpoint(request),
            mode="evaluation",
            is_training_data=request.is_training_data,
        )
        output_path = request.output_path or (request.run_dir / "evaluation.json")
        predictions_path = request.predictions_path or (request.run_dir / "validation_predictions.jsonl")
        evaluation = run.result

        labels = np.asarray(evaluation["labels"], dtype=np.float64)
        probabilities = np.asarray(evaluation["probabilities"], dtype=np.float64)
        metrics = compute_classification_metrics(
            labels,
            probabilities,
            auc_bootstrap_samples=_EVAL_AUC_BOOTSTRAP_SAMPLES,
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("wb") as handle:
            for record in evaluation["records"]:
                handle.write(dump_bytes(record))
                handle.write(b"\n")
        payload = {
            "experiment_name": self.name,
            "checkpoint_path": str(run.checkpoint),
            "schema_path": str(run.schema_path),
            "schema": run.schema,
            "metrics": metrics,
            "data_diagnostics": build_evaluation_data_diagnostics(run.dataset_path),
            "validation_predictions_path": str(predictions_path),
            "batch_size": run.batch_size,
            "num_workers": run.num_workers,
        }
        telemetry_payload = run.telemetry.finish(
            rows=int(evaluation.get("processed_rows", len(probabilities))),
            batches=int(evaluation.get("batch_count", 0) or 0),
            prediction_file_mb=file_size_mb(predictions_path),
            checkpoint_file_mb=file_size_mb(run.checkpoint),
        )
        payload["telemetry"] = telemetry_payload
        observed_schema_path = output_path.with_name("evaluation_observed_schema.json")
        write_observed_schema_report(
            dataset_path=run.dataset_path,
            schema_path=run.schema_path,
            output_path=observed_schema_path,
            dataset_role="eval",
        )
        payload["observed_schema_paths"] = {"eval": str(observed_schema_path)}
        write_json(output_path, payload)
        write_json(output_path.with_name("evaluation_telemetry.json"), telemetry_payload)
        return payload

    def infer(self, request: InferRequest) -> Mapping[str, Any]:
        run = self._execute_prediction_run(
            request,
            checkpoint=resolve_inference_checkpoint(request),
            mode="inference",
            is_training_data=False,
        )
        evaluation = run.result

        raw_predictions = evaluation.get("predictions")
        if raw_predictions is None:
            prediction_map = {
                str(record["user_id"]): float(record["score"])
                for record in evaluation["records"]
            }
        else:
            prediction_map = {
                str(user_id): float(score)
                for user_id, score in raw_predictions.items()
            }
        request.result_dir.mkdir(parents=True, exist_ok=True)
        output_path = request.result_dir / "predictions.json"
        write_json(output_path, {"predictions": prediction_map})
        telemetry_payload = run.telemetry.finish(
            rows=int(evaluation.get("processed_rows", len(prediction_map))),
            batches=int(evaluation.get("batch_count", 0) or 0),
            prediction_file_mb=file_size_mb(output_path),
            checkpoint_file_mb=file_size_mb(run.checkpoint),
        )
        write_json(request.result_dir / "inference_telemetry.json", telemetry_payload)
        return {
            "checkpoint_path": str(run.checkpoint),
            "schema_path": str(run.schema_path),
            "schema": run.schema,
            "predictions_path": str(output_path),
            "prediction_count": len(prediction_map),
            "batch_size": run.batch_size,
            "num_workers": run.num_workers,
            "telemetry": telemetry_payload,
        }


__all__ = ["PCVRExperiment", "_log_prediction_progress", "create_pcvr_experiment"]
