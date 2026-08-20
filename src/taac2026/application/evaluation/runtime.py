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


def resolve_evaluation_checkpoint(request: EvalRequest) -> Path:
    return resolve_checkpoint_path(request.run_dir, request.checkpoint_path)


def resolve_inference_checkpoint(request: InferRequest) -> Path:
    checkpoint_root = Path(os.environ.get("MODEL_OUTPUT_PATH", "")).expanduser()
    checkpoint = resolve_checkpoint_path(Path.cwd(), request.checkpoint_path) if request.checkpoint_path else None
    if checkpoint is None and str(checkpoint_root) not in {"", "."} and checkpoint_root.exists():
        checkpoint = resolve_checkpoint_path(checkpoint_root)
    if checkpoint is None:
        checkpoint = resolve_checkpoint_path(Path.cwd())
    return checkpoint


def load_train_config(config_type: type[PCVRTrainConfig], checkpoint_dir: Path) -> PCVRTrainConfig:
    config_path = checkpoint_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"PCVR train_config.json not found in checkpoint directory: {checkpoint_dir}")
    payload = load_pcvr_train_config_sidecar(read_json(config_path), config_type=config_type)
    return payload


def load_runtime_schema(
    *,
    schema_path: Path | None,
    checkpoint_dir: Path,
    mode: str,
) -> tuple[Path, Any]:
    resolved_schema_path = resolve_checkpoint_schema_path(checkpoint_dir, schema_path)
    logger.info("Resolved PCVR {} schema.json: {}", mode, resolved_schema_path)
    return resolved_schema_path, read_json(resolved_schema_path)


def build_evaluation_data_diagnostics(dataset_path: Path) -> dict[str, Any]:
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


def write_observed_schema_report(
    *,
    dataset_path: Path,
    schema_path: Path,
    output_path: Path,
    dataset_role: str,
    row_group_range: tuple[int, int] | None = None,
    timestamp_range: pcvr_data.PCVRTimestampRange | None = None,
    hash_split_filter: pcvr_data.PCVRHashSplitFilter | None = None,
) -> Path:
    report = pcvr_data.build_pcvr_observed_schema_report(
        dataset_path,
        schema_path,
        row_group_range=row_group_range,
        timestamp_range=timestamp_range,
        hash_split_filter=hash_split_filter,
        dataset_role=dataset_role,
    )
    write_json(output_path, report)
    logger.info("Wrote PCVR observed schema report for {}: {}", dataset_role, output_path)
    return output_path


def write_train_split_observed_schema_reports(
    *,
    dataset_path: Path,
    schema_path: Path,
    run_dir: Path,
    valid_ratio: float,
    train_ratio: float,
    split_strategy: str = "row_group_tail",
    split_seed: int = 42,
) -> dict[str, Any]:
    rg_info = pcvr_data.collect_pcvr_row_groups(dataset_path)
    split_plan, train_timestamp_range, valid_timestamp_range, train_hash_filter, valid_hash_filter = (
        pcvr_data.resolve_pcvr_split_plan(
            rg_info,
            split_strategy=split_strategy,
            valid_ratio=valid_ratio,
            train_ratio=train_ratio,
            seed=split_seed,
        )
    )
    observed_schema_paths = {
        "train_split": str(
            write_observed_schema_report(
                dataset_path=dataset_path,
                schema_path=schema_path,
                output_path=run_dir / "train_split_observed_schema.json",
                dataset_role="train_split",
                row_group_range=split_plan.train_row_group_range,
                timestamp_range=train_timestamp_range,
                hash_split_filter=train_hash_filter,
            )
        ),
        "valid_split": str(
            write_observed_schema_report(
                dataset_path=dataset_path,
                schema_path=schema_path,
                output_path=run_dir / "valid_split_observed_schema.json",
                dataset_role="valid_split",
                row_group_range=split_plan.valid_row_group_range,
                timestamp_range=valid_timestamp_range,
                hash_split_filter=valid_hash_filter,
            )
        ),
    }
    train_report = read_json(Path(observed_schema_paths["train_split"]))
    valid_report = read_json(Path(observed_schema_paths["valid_split"]))
    data_split = {
        "split_strategy": split_strategy,
        "train_timestamp_range": pcvr_data.pcvr_timestamp_range_to_dict(
            train_timestamp_range,
        ),
        "valid_timestamp_range": pcvr_data.pcvr_timestamp_range_to_dict(
            valid_timestamp_range,
        ),
    }
    if train_hash_filter is not None and valid_hash_filter is not None:
        data_split["train_hash_filter"] = train_report["hash_split_filter"]
        data_split["valid_hash_filter"] = valid_report["hash_split_filter"]
    train_rows = int(train_report["row_count"])
    valid_rows = int(valid_report["row_count"])
    uses_row_filter = (
        train_timestamp_range is not None
        or valid_timestamp_range is not None
        or train_hash_filter is not None
        or valid_hash_filter is not None
    )
    is_disjoint = True if uses_row_filter else split_plan.is_disjoint
    is_l1_ready = is_disjoint and train_rows > 0 and valid_rows > 0
    data_split["is_disjoint"] = is_disjoint
    data_split["is_l1_ready"] = is_l1_ready
    return {
        "observed_schema_paths": observed_schema_paths,
        "data_split": data_split,
        "row_group_split": {
            "train_row_groups": split_plan.train_row_groups,
            "valid_row_groups": split_plan.valid_row_groups,
            "train_row_group_range": list(split_plan.train_row_group_range),
            "valid_row_group_range": list(split_plan.valid_row_group_range),
            "train_rows": train_rows,
            "valid_rows": valid_rows,
            "reuse_train_for_valid": split_plan.reuse_train_for_valid,
            "is_disjoint": is_disjoint,
            "is_l1_ready": is_l1_ready,
        },
    }


__all__ = [
    "build_evaluation_data_diagnostics",
    "load_runtime_schema",
    "load_train_config",
    "resolve_evaluation_checkpoint",
    "resolve_inference_checkpoint",
    "write_observed_schema_report",
    "write_train_split_observed_schema_reports",
]
