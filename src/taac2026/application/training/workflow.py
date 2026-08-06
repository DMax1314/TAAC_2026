"""PCVR training workflow: data, model, trainer, and reporter preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

import taac2026.infrastructure.data.dataset as pcvr_data
from taac2026.domain.config import PCVRTrainConfig
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.modeling.model_contract import parse_seq_max_lens
from taac2026.infrastructure.logging import logger
from taac2026.infrastructure.modeling.sequence import configure_flash_attention_runtime as configure_shared_flash_attention_runtime
from taac2026.infrastructure.runtime.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.runtime.execution import EarlyStopping
from taac2026.infrastructure.checkpoints import preferred_checkpoint_path
from taac2026.infrastructure import modeling as shared_modeling


@dataclass(slots=True)
class PCVRTrainContext:
    model_module: Any
    model_class_name: str
    model_type: type[torch.nn.Module]
    package_dir: Path
    config: PCVRTrainConfig
    data_dir: Path
    ckpt_dir: Path
    log_dir: Path
    tf_events_dir: Path
    schema_path: Path
    runtime_execution: RuntimeExecutionConfig
    device: str
    reporter: Any

    @property
    def data_pipeline_config(self):
        return self.config.data_pipeline


@dataclass(frozen=True, slots=True)
class PCVRTrainDataBundle:
    train_loader: Any
    valid_loader: Any
    dataset: Any
    data_module: Any = pcvr_data


def default_build_train_data(context: PCVRTrainContext) -> PCVRTrainDataBundle:
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
        data_pipeline_config=context.data_pipeline_config,
        max_steps=context.config.optimizer.max_steps,
    )
    return PCVRTrainDataBundle(
        train_loader=train_loader,
        valid_loader=valid_loader,
        dataset=dataset,
    )


def default_build_train_model(
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


def default_build_train_trainer(
    context: PCVRTrainContext,
    data_bundle: PCVRTrainDataBundle,
    model: torch.nn.Module,
) -> Any:
    config = context.config
    early_stopping = EarlyStopping(
        checkpoint_path=preferred_checkpoint_path(context.ckpt_dir),
        patience_steps=config.optimizer.patience_steps,
        label="model",
    )
    return PCVRPointwiseTrainer(
        model=model,
        train_loader=data_bundle.train_loader,
        valid_loader=data_bundle.valid_loader,
        lr=config.optimizer.lr,
        max_steps=config.optimizer.max_steps,
        device=context.device,
        save_dir=context.ckpt_dir,
        early_stopping=early_stopping,
        dense_optimizer_type=config.optimizer.dense_optimizer_type,
        scheduler_type=config.optimizer.scheduler_type,
        warmup_steps=config.optimizer.warmup_steps,
        min_lr_ratio=config.optimizer.min_lr_ratio,
        ema_enabled=config.ema.enabled,
        ema_decay=config.ema.decay,
        ema_start_step=config.ema.start_step,
        ema_update_every_n_steps=config.ema.update_every_n_steps,
        loss_terms=config.loss.to_list(),
        sparse_lr=config.sparse_optimizer.sparse_lr,
        sparse_weight_decay=config.sparse_optimizer.sparse_weight_decay,
        reinit_sparse_every_n_steps=config.sparse_optimizer.reinit_sparse_every_n_steps,
        reinit_cardinality_threshold=config.sparse_optimizer.reinit_cardinality_threshold,
        reporter=context.reporter,
        schema_path=context.schema_path,
        eval_every_n_steps=config.data.eval_every_n_steps,
        early_stopping_metric=config.validation.early_stopping_metric,
        train_config=config,
        runtime_execution=context.runtime_execution,
    )


def default_run_training(context: PCVRTrainContext, trainer: Any) -> None:
    del context
    trainer.train()


def default_build_train_summary(
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
    return summary


class TrainReporter(Protocol):
    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None:
        ...

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
    ) -> None:
        ...

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool:
        ...

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        ...

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        ...

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        ...

    def close(self) -> None:
        ...


class NoopTrainReporter:
    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None:
        pass

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
    ) -> None:
        pass

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool:
        return False

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        pass

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        return {}

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardTrainReporter:
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None:
        self.writer.add_scalar("Loss/train", float(loss), int(step))
        for name, value in loss_components.items():
            self.writer.add_scalar(f"Loss/train/{name}", float(value), int(step))
        self.writer.add_scalar("LR/dense", float(dense_lr), int(step))

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
    ) -> None:
        del metrics
        self.writer.add_scalar("AUC/valid", float(auc), int(step))
        self.writer.add_scalar("LogLoss/valid", float(logloss), int(step))
        for metric_name, value in score_diagnostics.items():
            self.writer.add_scalar(f"score/{metric_name}", float(value), int(step))
        self.writer.flush()

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool:
        del phase
        if step is None:
            return False
        interval = int(trainer.runtime_execution.progress_log_interval_steps)
        return (
            step == 1
            or (interval > 0 and step % interval == 0)
            or (trainer.eval_every_n_steps > 0 and step % trainer.eval_every_n_steps == 0)
            or (trainer.max_steps > 0 and step == trainer.max_steps)
        )

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None:
        set_enabled = getattr(model, "set_training_diagnostics_enabled", None)
        if not callable(set_enabled):
            set_enabled = getattr(model, "set_tensorboard_diagnostics_enabled", None)
        if callable(set_enabled):
            set_enabled(enabled)

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        consume_scalars = getattr(model, "consume_training_scalars", None)
        if not callable(consume_scalars):
            consume_scalars = getattr(model, "consume_tensorboard_scalars", None)
        if not callable(consume_scalars):
            return {}
        return consume_scalars(phase=phase)

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        del phase
        for tag, value in scalars.items():
            self.writer.add_scalar(str(tag), float(value), int(step))

    def close(self) -> None:
        self.writer.close()


def default_build_train_reporter(context: PCVRTrainContext) -> TrainReporter:
    return TensorBoardTrainReporter(context.tf_events_dir)


__all__ = [
    "NoopTrainReporter",
    "PCVRTrainContext",
    "PCVRTrainDataBundle",
    "TensorBoardTrainReporter",
    "TrainReporter",
    "default_build_train_data",
    "default_build_train_model",
    "default_build_train_reporter",
    "default_build_train_summary",
    "default_build_train_trainer",
    "default_run_training",
]
