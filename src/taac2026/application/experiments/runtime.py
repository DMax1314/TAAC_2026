"""Runtime helpers extracted from the PCVR experiment adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from taac2026.domain.requests import EvalRequest, InferRequest
from taac2026.domain.config import PCVRTrainConfig
from taac2026.domain.runtime_config import RuntimeExecutionConfig, normalize_amp_dtype
from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    default_build_prediction_data,
    default_build_prediction_model,
    default_prepare_prediction_runner,
    default_run_prediction_loop,
)
from taac2026.application.evaluation.runtime import (
    default_load_train_config,
    default_load_runtime_schema,
    default_write_observed_schema_report,
    default_write_train_split_observed_schema_reports,
)
from taac2026.infrastructure.experiments.module_loader import load_experiment_submodule


_INFER_REQUEST_DEFAULT_BATCH_SIZE = int(InferRequest.__dataclass_fields__["batch_size"].default)
_INFER_REQUEST_DEFAULT_NUM_WORKERS = int(InferRequest.__dataclass_fields__["num_workers"].default)


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class PCVRExperimentRuntimeMixin:
    def _load_model_module(self) -> Any:
        return load_experiment_submodule(self.package_dir, "model")

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
            configured_batch_size, configured_batch_size_source = self._configured_infer_runtime_value(
                config,
                config_key="batch_size",
                minimum=1,
            )
            batch_size = configured_batch_size
            batch_size_source = configured_batch_size_source

        num_workers = int(request.num_workers)
        num_workers_source = "request" if request.num_workers != _INFER_REQUEST_DEFAULT_NUM_WORKERS else "cli_default"
        if num_workers_source == "cli_default":
            configured_num_workers, configured_num_workers_source = self._configured_infer_runtime_value(
                config,
                config_key="num_workers",
                minimum=0,
            )
            num_workers = configured_num_workers
            num_workers_source = configured_num_workers_source

        return batch_size, batch_size_source, num_workers, num_workers_source

    def _resolve_prediction_runtime_execution(
        self,
        request: EvalRequest | InferRequest,
        config: PCVRTrainConfig,
    ) -> tuple[RuntimeExecutionConfig, str, str, str]:
        runtime = config.runtime
        amp = runtime.amp if getattr(request, "amp", None) is None else bool(request.amp)
        amp_source = "train_config" if getattr(request, "amp", None) is None else "request"
        amp_dtype = runtime.amp_dtype if getattr(request, "amp_dtype", None) in (None, "") else normalize_amp_dtype(request.amp_dtype)
        amp_dtype_source = "train_config" if getattr(request, "amp_dtype", None) in (None, "") else "request"
        compile_enabled = runtime.compile if getattr(request, "compile", None) is None else bool(request.compile)
        compile_source = "train_config" if getattr(request, "compile", None) is None else "request"
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
        model_module = self._load_model_module()

        resolved_schema_path, _resolved_schema = default_load_runtime_schema(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_dir=checkpoint_path.parent,
            mode="evaluation" if is_training_data else "inference",
        )
        resolved_config = config if config is not None else default_load_train_config(self, checkpoint_path.parent)
        resolved_runtime_execution = runtime_execution or RuntimeExecutionConfig()
        context = PCVRPredictionContext(
            model_module=model_module,
            model_class_name=self.model_class_name,
            model_type=self.model_type,
            package_dir=self.package_dir,
            dataset_path=dataset_path,
            schema_path=resolved_schema_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            is_training_data=is_training_data,
            dataset_role=dataset_role,
            config=resolved_config,
            runtime_execution=resolved_runtime_execution,
        )
        data_bundle = default_build_prediction_data(context)
        model = default_build_prediction_model(context, data_bundle)
        runner = default_prepare_prediction_runner(context, data_bundle, model)
        return default_run_prediction_loop(context, data_bundle, runner)

    def _load_train_config(self, checkpoint_dir: Path) -> PCVRTrainConfig:
        return default_load_train_config(self, checkpoint_dir)

    def _load_resolved_schema(
        self,
        *,
        dataset_path: Path,
        schema_path: Path | None,
        checkpoint_dir: Path,
        mode: str,
    ) -> tuple[Path, Any]:
        return default_load_runtime_schema(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_dir=checkpoint_dir,
            mode=mode,
        )

    def _write_observed_schema_report(
        self,
        *,
        dataset_path: Path,
        schema_path: Path,
        output_path: Path,
        dataset_role: str,
        row_group_range: tuple[int, int] | None = None,
        timestamp_range: Any = None,
    ) -> Path:
        return default_write_observed_schema_report(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            output_path=output_path,
            dataset_role=dataset_role,
            row_group_range=row_group_range,
            timestamp_range=timestamp_range,
        )

    def _write_train_split_observed_schema_reports(
        self,
        *,
        dataset_path: Path,
        schema_path: Path,
        run_dir: Path,
        valid_ratio: float,
        train_ratio: float,
        split_strategy: str = "row_group_tail",
    ) -> dict[str, Any]:
        return default_write_train_split_observed_schema_reports(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            run_dir=run_dir,
            valid_ratio=valid_ratio,
            train_ratio=train_ratio,
            split_strategy=split_strategy,
        )

    def _resolve_schema_path(self, dataset_path: Path, schema_path: Path | None, checkpoint_dir: Path) -> Path:
        resolved_schema_path, _schema_payload = default_load_runtime_schema(
            self,
            dataset_path=dataset_path,
            schema_path=schema_path,
            checkpoint_dir=checkpoint_dir,
            mode="runtime",
        )
        return resolved_schema_path


__all__ = ["PCVRExperimentRuntimeMixin"]
