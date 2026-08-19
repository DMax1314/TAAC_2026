"""Shared PCVR model training entrypoint."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import tyro

from taac2026.domain.config import PCVRTrainConfig
from taac2026.infrastructure.modeling.model_contract import resolve_training_schema_path
from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.logging import logger
from taac2026.application.training.workflow import (
    PCVRTrainContext,
    build_train_data,
    build_train_model,
    build_train_summary,
    build_train_trainer,
)
from taac2026.infrastructure.runtime.telemetry import RuntimeTelemetry
from taac2026.infrastructure.runtime.reporting import TensorBoardTrainReporter
from taac2026.infrastructure.runtime.execution import (
    create_logger,
    runtime_execution_summary,
    set_seed,
)


def parse_pcvr_train_config(
    argv: Sequence[str] | None,
    *,
    config_type: type[PCVRTrainConfig],
    defaults: PCVRTrainConfig,
) -> PCVRTrainConfig:
    """Parse the typed nested experiment config from CLI arguments, layered on defaults."""
    return tyro.cli(
        config_type,
        description="Train a PCVR experiment",
        args=argv,
        default=defaults,
        use_underscores=True,
    )


def train_pcvr_model(
    *,
    model_class_name: str,
    defaults: PCVRTrainConfig,
    config_type: type[PCVRTrainConfig],
    model_type: type[torch.nn.Module],
    argv: Sequence[str] | None,
    dataset_path: Path,
    schema_path_override: Path | None,
    ckpt_dir: Path,
    log_dir: Path,
    tf_events_dir: Path,
) -> dict[str, Any]:
    config = parse_pcvr_train_config(argv, config_type=config_type, defaults=defaults)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tf_events_dir.mkdir(parents=True, exist_ok=True)

    deterministic = bool(config.runtime.deterministic)
    set_seed(config.optimizer.seed, deterministic=deterministic)
    create_logger(log_dir / "train.log")
    device = config.optimizer.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Args: {}", config.model_dump(mode="json"))
    logger.info(
        "Resolved PCVR training runtime: {}", runtime_execution_summary(config.runtime, device)
    )

    reporter = None
    telemetry = RuntimeTelemetry(
        label="training",
        device=device,
        metadata={
            "model_class": model_class_name,
            "run_dir": str(ckpt_dir),
        },
    ).start()
    try:
        schema_path = resolve_training_schema_path(dataset_path, schema_path_override)
        context = PCVRTrainContext(
            model_class_name=model_class_name,
            model_type=model_type,
            config=config,
            data_dir=dataset_path,
            ckpt_dir=ckpt_dir,
            schema_path=schema_path,
            device=device,
            reporter=TensorBoardTrainReporter(tf_events_dir),
        )
        reporter = context.reporter
        data_bundle = build_train_data(context)
        model = build_train_model(context, data_bundle)
        trainer = build_train_trainer(context, data_bundle, model)
        trainer.train()
        summary = dict(build_train_summary(context, trainer) or {})
        train_rows = int(getattr(data_bundle.dataset, "num_rows", 0) or 0)
        valid_dataset = getattr(data_bundle.valid_loader, "dataset", None)
        valid_rows = int(getattr(valid_dataset, "num_rows", 0) or 0)
        summary["telemetry"] = telemetry.finish(
            steps=int(getattr(trainer, "optim_step", 0) or 0),
            train_rows=train_rows,
            valid_rows=valid_rows,
            rows=train_rows,
            model_parameters=int(sum(parameter.numel() for parameter in model.parameters()))
            if hasattr(model, "parameters")
            else 0,
        )
        write_json(ckpt_dir / "training_telemetry.json", summary["telemetry"])
        write_json(ckpt_dir / "training_summary.json", summary)
    finally:
        if reporter is not None:
            reporter.close()

    logger.info("Training complete!")
    return summary
