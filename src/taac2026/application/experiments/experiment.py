"""PCVR experiment adapter for plugin packages."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest
from taac2026.domain.metrics import compute_classification_metrics
from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.io.json import dump_bytes
from taac2026.domain.config import PCVRTrainConfig
from taac2026.application.experiments.runtime import PCVRExperimentRuntimeMixin
from taac2026.application.evaluation.workflow import _log_prediction_progress
from taac2026.application.evaluation.runtime import (
    default_build_evaluation_data_diagnostics,
    default_load_train_config,
    default_load_runtime_schema,
    default_resolve_evaluation_checkpoint,
    default_resolve_inference_checkpoint,
    default_write_observed_schema_report,
    default_write_train_split_observed_schema_reports,
)
from taac2026.infrastructure.data.sample_dataset import resolve_default_pcvr_sample_paths
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.runtime.telemetry import RuntimeTelemetry, file_size_mb
from taac2026.application.training.args import train_pcvr_model


_EVAL_AUC_BOOTSTRAP_SAMPLES = 200


def _callable_name(value: Any) -> str:
    return getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))


@dataclass(slots=True)
class PCVRExperiment(PCVRExperimentRuntimeMixin):
    name: str
    package_dir: Path
    model_type: type[torch.nn.Module]
    config_type: type[PCVRTrainConfig]
    train_defaults: PCVRTrainConfig

    @property
    def model_class_name(self) -> str:
        return self.model_type.__name__

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "kind": "pcvr",
            "model_class": self.model_class_name,
            "source": str(self.package_dir),
            "config_type": _callable_name(self.config_type),
        }

    @contextmanager
    def _module_context(self) -> Iterator[None]:
        yield

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

        with self._module_context():
            model_module = self._load_model_module()

            summary = dict(train_pcvr_model(
                model_module=model_module,
                model_class_name=self.model_class_name,
                model_type=self.model_type,
                package_dir=self.package_dir,
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

        observed_schema_payload = default_write_train_split_observed_schema_reports(
            self,
            dataset_path=dataset_path,
            schema_path=resolved_schema_path,
            run_dir=run_dir,
            valid_ratio=float(summary["valid_ratio"]),
            train_ratio=float(summary["train_ratio"]),
            split_strategy=str(summary.get("split_strategy", "row_group_tail")),
        )

        payload = dict(summary)
        payload["experiment_name"] = self.name
        payload["run_dir"] = str(run_dir)
        payload["checkpoint_root"] = str(ckpt_dir)
        payload["schema_path"] = str(resolved_schema_path)
        payload.update(observed_schema_payload)
        return payload

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]:
        resolved_dataset_path, resolved_schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        checkpoint = default_resolve_evaluation_checkpoint(self, request)
        output_path = request.output_path or (request.run_dir / "evaluation.json")
        predictions_path = request.predictions_path or (request.run_dir / "validation_predictions.jsonl")
        config = default_load_train_config(self, checkpoint.parent)
        resolved_schema_path, resolved_schema = default_load_runtime_schema(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_override,
            checkpoint_dir=checkpoint.parent,
            mode="evaluation",
        )
        effective_batch_size, batch_size_source, effective_num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logger.info(
            "Resolved PCVR evaluation runtime: experiment={}, checkpoint={}, batch_size={} ({}), num_workers={} ({}), amp={} ({}), amp_dtype={} ({}), compile={} ({})",
            self.name,
            checkpoint,
            effective_batch_size,
            batch_size_source,
            effective_num_workers,
            num_workers_source,
            runtime_execution.amp,
            amp_source,
            runtime_execution.normalized_amp_dtype(),
            amp_dtype_source,
            runtime_execution.compile,
            compile_source,
        )

        telemetry = RuntimeTelemetry(
            label="evaluation",
            device=request.device,
            metadata={
                "experiment_name": self.name,
                "checkpoint_path": str(checkpoint),
                "dataset_path": str(resolved_dataset_path),
            },
        ).start()
        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=resolved_dataset_path,
                schema_path=resolved_schema_path,
                checkpoint_path=checkpoint,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=request.device,
                is_training_data=request.is_training_data,
                dataset_role="evaluation",
                config=config,
                runtime_execution=runtime_execution,
            )

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
            "checkpoint_path": str(checkpoint),
            "schema_path": str(resolved_schema_path),
            "schema": resolved_schema,
            "metrics": metrics,
            "data_diagnostics": default_build_evaluation_data_diagnostics(self, resolved_dataset_path),
            "validation_predictions_path": str(predictions_path),
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
        }
        telemetry_payload = telemetry.finish(
            rows=int(evaluation.get("processed_rows", len(probabilities))),
            batches=int(evaluation.get("batch_count", 0) or 0),
            prediction_file_mb=file_size_mb(predictions_path),
            checkpoint_file_mb=file_size_mb(checkpoint),
        )
        payload["telemetry"] = telemetry_payload
        observed_schema_path = output_path.with_name("evaluation_observed_schema.json")
        default_write_observed_schema_report(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_path,
            output_path=observed_schema_path,
            dataset_role="eval",
        )
        payload["observed_schema_paths"] = {"eval": str(observed_schema_path)}
        write_json(output_path, payload)
        write_json(output_path.with_name("evaluation_telemetry.json"), telemetry_payload)
        return payload

    def infer(self, request: InferRequest) -> Mapping[str, Any]:
        resolved_dataset_path, resolved_schema_override = resolve_default_pcvr_sample_paths(
            request.dataset_path,
            request.schema_path,
        )
        checkpoint = default_resolve_inference_checkpoint(self, request)
        config = default_load_train_config(self, checkpoint.parent)
        resolved_schema_path, resolved_schema = default_load_runtime_schema(
            self,
            dataset_path=resolved_dataset_path,
            schema_path=resolved_schema_override,
            checkpoint_dir=checkpoint.parent,
            mode="inference",
        )
        effective_batch_size, batch_size_source, effective_num_workers, num_workers_source = self._resolve_prediction_runtime_settings(
            request,
            config,
        )
        runtime_execution, amp_source, amp_dtype_source, compile_source = self._resolve_prediction_runtime_execution(
            request,
            config,
        )
        logger.info(
            "Resolved PCVR inference runtime: experiment={}, checkpoint={}, batch_size={} ({}), num_workers={} ({}), amp={} ({}), amp_dtype={} ({}), compile={} ({})",
            self.name,
            checkpoint,
            effective_batch_size,
            batch_size_source,
            effective_num_workers,
            num_workers_source,
            runtime_execution.amp,
            amp_source,
            runtime_execution.normalized_amp_dtype(),
            amp_dtype_source,
            runtime_execution.compile,
            compile_source,
        )

        telemetry = RuntimeTelemetry(
            label="inference",
            device=request.device,
            metadata={
                "experiment_name": self.name,
                "checkpoint_path": str(checkpoint),
                "dataset_path": str(resolved_dataset_path),
            },
        ).start()
        with self._module_context():
            evaluation = self._run_prediction_loop(
                dataset_path=resolved_dataset_path,
                schema_path=resolved_schema_path,
                checkpoint_path=checkpoint,
                batch_size=effective_batch_size,
                num_workers=effective_num_workers,
                device=request.device,
                is_training_data=False,
                dataset_role="inference",
                config=config,
                runtime_execution=runtime_execution,
            )

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
        telemetry_payload = telemetry.finish(
            rows=int(evaluation.get("processed_rows", len(prediction_map))),
            batches=int(evaluation.get("batch_count", 0) or 0),
            prediction_file_mb=file_size_mb(output_path),
            checkpoint_file_mb=file_size_mb(checkpoint),
        )
        write_json(request.result_dir / "inference_telemetry.json", telemetry_payload)
        return {
            "checkpoint_path": str(checkpoint),
            "schema_path": str(resolved_schema_path),
            "schema": resolved_schema,
            "predictions_path": str(output_path),
            "prediction_count": len(prediction_map),
            "batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
            "telemetry": telemetry_payload,
        }


__all__ = ["PCVRExperiment", "_log_prediction_progress"]
