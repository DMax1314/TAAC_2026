"""PCVR evaluation runtime services for checkpoint, schema, and report handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from taac2026.domain.requests import EvalRequest, InferRequest
from taac2026.infrastructure.checkpoints import resolve_checkpoint_path
from taac2026.infrastructure.io.files import read_json, write_json
import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.config import (
    PCVRTrainConfig,
)
from taac2026.infrastructure.logging import logger
from taac2026.domain.sidecar import load_pcvr_train_config_sidecar
from taac2026.infrastructure.modeling.model_contract import resolve_checkpoint_schema_path


def default_resolve_evaluation_checkpoint(experiment: Any, request: EvalRequest) -> Path:
    del experiment
    return resolve_checkpoint_path(request.run_dir, request.checkpoint_path)


def default_resolve_inference_checkpoint(experiment: Any, request: InferRequest) -> Path:
    del experiment
    checkpoint_root = Path(os.environ.get("MODEL_OUTPUT_PATH", "")).expanduser()
    checkpoint = resolve_checkpoint_path(Path.cwd(), request.checkpoint_path) if request.checkpoint_path else None
    if checkpoint is None and str(checkpoint_root) not in {"", "."} and checkpoint_root.exists():
        checkpoint = resolve_checkpoint_path(checkpoint_root)
    if checkpoint is None:
        checkpoint = resolve_checkpoint_path(Path.cwd())
    return checkpoint


def default_load_train_config(experiment: Any, checkpoint_dir: Path) -> PCVRTrainConfig:
    config_path = checkpoint_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"PCVR train_config.json not found in checkpoint directory: {checkpoint_dir}")
    payload = load_pcvr_train_config_sidecar(read_json(config_path), config_type=experiment.config_type)
    return payload


def default_load_runtime_schema(
    experiment: Any,
    *,
    dataset_path: Path,
    schema_path: Path | None,
    checkpoint_dir: Path,
    mode: str,
) -> tuple[Path, Any]:
    del experiment, dataset_path
    resolved_schema_path = resolve_checkpoint_schema_path(checkpoint_dir, schema_path)
    logger.info("Resolved PCVR {} schema.json: {}", mode, resolved_schema_path)
    return resolved_schema_path, read_json(resolved_schema_path)


def default_build_evaluation_data_diagnostics(experiment: Any, dataset_path: Path) -> dict[str, Any]:
    del experiment
    resolved_dataset_path = dataset_path.expanduser()
    warnings: list[str] = []
    try:
        rg_info = pcvr_data.collect_pcvr_row_groups(resolved_dataset_path)
        split_plan = pcvr_data.plan_pcvr_row_group_split(rg_info)
    except (FileNotFoundError, OSError, ValueError) as error:
        return {
            "dataset_path": str(resolved_dataset_path),
            "warnings": [f"row group diagnostics unavailable: {error}"],
        }

    files = sorted({path for path, _index, _rows in rg_info})
    if split_plan.reuse_train_for_valid:
        warnings.append("single Row Group dataset would reuse train rows for validation; treat as L0 smoke only")
    if not split_plan.is_l1_ready:
        warnings.append("row group split is not suitable for L1 model comparison")

    return {
        "dataset_path": str(resolved_dataset_path.resolve()),
        "file_count": len(files),
        "total_row_groups": split_plan.total_row_groups,
        "total_rows": int(sum(rows for _path, _index, rows in rg_info)),
        "row_group_split": {
            "train_row_groups": split_plan.train_row_groups,
            "valid_row_groups": split_plan.valid_row_groups,
            "train_row_group_range": list(split_plan.train_row_group_range),
            "valid_row_group_range": list(split_plan.valid_row_group_range),
            "train_rows": split_plan.train_rows,
            "valid_rows": split_plan.valid_rows,
            "reuse_train_for_valid": split_plan.reuse_train_for_valid,
            "is_disjoint": split_plan.is_disjoint,
            "is_l1_ready": split_plan.is_l1_ready,
        },
        "warnings": warnings,
    }


def default_write_observed_schema_report(
    experiment: Any,
    *,
    dataset_path: Path,
    schema_path: Path,
    output_path: Path,
    dataset_role: str,
    row_group_range: tuple[int, int] | None = None,
    timestamp_range: pcvr_data.PCVRTimestampRange | None = None,
) -> Path:
    report = pcvr_data.build_pcvr_observed_schema_report(
        dataset_path,
        schema_path,
        row_group_range=row_group_range,
        timestamp_range=timestamp_range,
        dataset_role=dataset_role,
    )
    write_json(output_path, report)
    logger.info("Wrote PCVR observed schema report for {}: {}", dataset_role, output_path)
    return output_path


def default_write_train_split_observed_schema_reports(
    experiment: Any,
    *,
    dataset_path: Path,
    schema_path: Path,
    run_dir: Path,
    valid_ratio: float,
    train_ratio: float,
    split_strategy: str = "row_group_tail",
) -> dict[str, Any]:
    rg_info = pcvr_data.collect_pcvr_row_groups(dataset_path)
    split_plan = pcvr_data.plan_pcvr_row_group_split(
        rg_info,
        valid_ratio=valid_ratio,
        train_ratio=train_ratio,
    )
    train_timestamp_range = None
    valid_timestamp_range = None
    if split_strategy == "timestamp_auto":
        split_plan, train_timestamp_range, valid_timestamp_range = (
            pcvr_data.plan_pcvr_timestamp_tail_split(
                rg_info,
                valid_ratio=valid_ratio,
                train_ratio=train_ratio,
            )
        )
    elif split_strategy != "row_group_tail":
        raise ValueError(f"unsupported split_strategy={split_strategy!r}")
    observed_schema_paths = {
        "train_split": str(
            default_write_observed_schema_report(
                experiment,
                dataset_path=dataset_path,
                schema_path=schema_path,
                output_path=run_dir / "train_split_observed_schema.json",
                dataset_role="train_split",
                row_group_range=split_plan.train_row_group_range,
                timestamp_range=train_timestamp_range,
            )
        ),
        "valid_split": str(
            default_write_observed_schema_report(
                experiment,
                dataset_path=dataset_path,
                schema_path=schema_path,
                output_path=run_dir / "valid_split_observed_schema.json",
                dataset_role="valid_split",
                row_group_range=split_plan.valid_row_group_range,
                timestamp_range=valid_timestamp_range,
            )
        ),
    }
    return {
        "observed_schema_paths": observed_schema_paths,
        "data_split": {
            "split_strategy": split_strategy,
            "train_timestamp_range": pcvr_data.pcvr_timestamp_range_to_dict(
                train_timestamp_range,
            ),
            "valid_timestamp_range": pcvr_data.pcvr_timestamp_range_to_dict(
                valid_timestamp_range,
            ),
        },
        "row_group_split": {
            "train_row_groups": split_plan.train_row_groups,
            "valid_row_groups": split_plan.valid_row_groups,
            "train_row_group_range": list(split_plan.train_row_group_range),
            "valid_row_group_range": list(split_plan.valid_row_group_range),
            "train_rows": split_plan.train_rows,
            "valid_rows": split_plan.valid_rows,
            "reuse_train_for_valid": split_plan.reuse_train_for_valid,
            "is_disjoint": split_plan.is_disjoint,
            "is_l1_ready": split_plan.is_l1_ready,
        },
    }


__all__ = [
    "default_build_evaluation_data_diagnostics",
    "default_load_runtime_schema",
    "default_load_train_config",
    "default_resolve_evaluation_checkpoint",
    "default_resolve_inference_checkpoint",
    "default_write_observed_schema_report",
    "default_write_train_split_observed_schema_reports",
]
