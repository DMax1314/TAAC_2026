"""PCVR training workflow: data, model, trainer, and reporter preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.config import PCVRTrainConfig
from taac2026.infrastructure.modeling.model_contract import parse_seq_max_lens
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.modeling.sequence import configure_flash_attention_runtime as configure_shared_flash_attention_runtime
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.runtime.reporting import TrainReporter
from taac2026.infrastructure import modeling as shared_modeling


@dataclass(slots=True)
class PCVRTrainContext:
    model_class_name: str
    model_type: type[torch.nn.Module]
    config: PCVRTrainConfig
    data_dir: Path
    ckpt_dir: Path
    schema_path: Path
    device: str
    reporter: TrainReporter


@dataclass(frozen=True, slots=True)
class PCVRTrainDataBundle:
    train_loader: Any
    valid_loader: Any
    dataset: Any


def build_train_data(context: PCVRTrainContext) -> PCVRTrainDataBundle:
    data_config = context.config.data
    seq_max_lens = parse_seq_max_lens(data_config.seq_max_lens)
    if seq_max_lens:
        logger.info("Seq max_lens override: {}", seq_max_lens)

    logger.info("Using PCVR train data pipeline: {}.get_pcvr_data", pcvr_data.__name__)
    train_loader, valid_loader, dataset = pcvr_data.get_pcvr_data(
        data_dir=str(context.data_dir),
        schema_path=str(context.schema_path),
        batch_size=data_config.batch_size,
        valid_ratio=data_config.valid_ratio,
        train_ratio=data_config.train_ratio,
        split_strategy=data_config.split_strategy,
        sampling_strategy=data_config.sampling_strategy,
        train_steps_per_sweep=data_config.train_steps_per_sweep,
        num_workers=data_config.num_workers,
        buffer_batches=data_config.buffer_batches,
        seed=context.config.optimizer.seed,
        seq_max_lens=seq_max_lens,
        data_pipeline_config=context.config.data_pipeline,
        max_steps=context.config.optimizer.max_steps,
    )
    return PCVRTrainDataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        dataset=dataset,
    )


def build_train_model(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
) -> torch.nn.Module:
    model_config = context.config.model
    configure_shared_flash_attention_runtime(backend=model_config.flash_attention_backend)
    shared_modeling.configure_rms_norm_runtime(
        backend=model_config.rms_norm_backend,
        block_rows=model_config.rms_norm_block_rows,
    )

    model = context.model_type(
        schema=data_bundle.dataset.layout.schema,
        config=model_config,
    ).to(context.device)

    num_sequences = len(data_bundle.dataset.seq_domains)
    num_ns = model.num_ns
    token_count = model_config.num_queries * num_sequences + num_ns
    logger.info(
        "PCVR model created: class={}, num_ns={}, T={}, d_model={}, rank_mixer_mode={}",
        context.model_class_name,
        num_ns,
        token_count,
        model_config.d_model,
        model_config.rank_mixer_mode,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Total parameters: {}", f"{total_params:,}")
    return model


def build_train_trainer(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
    model: torch.nn.Module,
) -> Any:
    return PCVRPointwiseTrainer(
        model=model,
        train_loader=data_bundle.train_loader,
        valid_loader=data_bundle.valid_loader,
        config=context.config,
        save_dir=context.ckpt_dir,
        reporter=context.reporter,
        schema_path=context.schema_path,
    )


def build_train_summary(
    context: PCVRTrainContext,
    trainer: Any,
) -> dict[str, Any]:
    data_config = context.config.data
    summary = {
        "run_dir": str(context.ckpt_dir),
        "checkpoint_root": str(context.ckpt_dir),
        "schema_path": str(context.schema_path),
        "train_ratio": float(data_config.train_ratio),
        "valid_ratio": float(data_config.valid_ratio),
        "split_strategy": str(data_config.split_strategy),
        "split_seed": int(context.config.optimizer.seed),
        "sampling_strategy": str(data_config.sampling_strategy),
        "train_steps_per_sweep": int(data_config.train_steps_per_sweep),
    }
    train_loader = getattr(trainer, "train_loader", None)
    train_dataset = getattr(train_loader, "dataset", None)
    pipeline = getattr(train_dataset, "pipeline", None)
    cache = getattr(pipeline, "cache", None)
    stats_fn = getattr(cache, "stats", None)
    if callable(stats_fn):
        summary["data_cache_stats"] = stats_fn()
    validation_metrics = dict(getattr(trainer, "last_eval_metrics", {}) or {})
    if validation_metrics:
        summary["validation_metrics"] = validation_metrics
    validation_score_diagnostics = dict(getattr(trainer, "last_eval_diagnostics", {}) or {})
    if validation_score_diagnostics:
        summary["validation_score_diagnostics"] = validation_score_diagnostics
    validation_model_scalars = dict(getattr(trainer, "last_eval_model_scalars", {}) or {})
    if validation_model_scalars:
        summary["validation_model_scalars"] = validation_model_scalars
    return summary


__all__ = [
    "PCVRTrainContext",
    "PCVRTrainDataBundle",
    "build_train_data",
    "build_train_model",
    "build_train_summary",
    "build_train_trainer",
]
