"""Training metric reporting contracts and implementations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch


class TrainReporter(Protocol):
    def train_step(self, *, step: int, loss: float, loss_components: Mapping[str, float], dense_lr: float) -> None: ...

    def validation_step(
        self,
        *,
        step: int,
        auc: float,
        logloss: float,
        metrics: Mapping[str, float],
        score_diagnostics: Mapping[str, float | int],
    ) -> None: ...

    def should_collect_model_scalars(self, *, phase: str, step: int | None, trainer: Any) -> bool: ...

    def set_model_diagnostics_enabled(self, model: torch.nn.Module, enabled: bool) -> None: ...

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]: ...

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None: ...

    def close(self) -> None: ...


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
        if callable(set_enabled):
            set_enabled(enabled)

    def consume_model_scalars(self, model: torch.nn.Module, *, phase: str) -> Mapping[str, float]:
        consume_scalars = getattr(model, "consume_training_scalars", None)
        if not callable(consume_scalars):
            return {}
        return consume_scalars(phase=phase)

    def model_scalars(self, *, phase: str, step: int, scalars: Mapping[str, float]) -> None:
        del phase
        for tag, value in scalars.items():
            self.writer.add_scalar(str(tag), float(value), int(step))

    def close(self) -> None:
        self.writer.close()


__all__ = ["NoopTrainReporter", "TensorBoardTrainReporter", "TrainReporter"]
