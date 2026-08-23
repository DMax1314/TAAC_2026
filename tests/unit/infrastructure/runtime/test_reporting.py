from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils import tensorboard as tensorboard_module

from taac2026.infrastructure.runtime.reporting import TensorBoardTrainReporter


class _FakeSummaryWriter:
    def __init__(self, _log_dir) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class _DiagnosticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enabled: list[bool] = []

    def set_training_diagnostics_enabled(self, enabled: bool) -> None:
        self.enabled.append(enabled)

    def consume_training_scalars(self, *, phase: str) -> dict[str, float]:
        return {f"Model/diagnostic/{phase}": 3.5}


def test_tensorboard_reporter_writes_runtime_metrics_and_model_diagnostics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tensorboard_module, "SummaryWriter", _FakeSummaryWriter)
    reporter = TensorBoardTrainReporter(tmp_path)
    model = _DiagnosticModel()

    reporter.train_step(step=2, loss=0.4, loss_components={"bce": 0.3}, dense_lr=1e-3)
    reporter.validation_step(
        step=2,
        auc=0.8,
        logloss=0.5,
        metrics={},
        score_diagnostics={"score_std": 0.1},
    )
    reporter.set_model_diagnostics_enabled(model, True)
    scalars = reporter.consume_model_scalars(model, phase="valid")
    reporter.model_scalars(phase="valid", step=2, scalars=scalars)
    reporter.close()

    assert reporter.writer.scalars == [
        ("Loss/train", 0.4, 2),
        ("Loss/train/bce", 0.3, 2),
        ("LR/dense", 1e-3, 2),
        ("AUC/valid", 0.8, 2),
        ("LogLoss/valid", 0.5, 2),
        ("score/score_std", 0.1, 2),
        ("Model/diagnostic/valid", 3.5, 2),
    ]
    assert reporter.writer.flush_count == 1
    assert reporter.writer.closed is True
    assert model.enabled == [True]


def test_tensorboard_reporter_collects_scalars_only_at_runtime_boundaries(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tensorboard_module, "SummaryWriter", _FakeSummaryWriter)
    reporter = TensorBoardTrainReporter(tmp_path)
    trainer = SimpleNamespace(
        runtime_execution=SimpleNamespace(progress_log_interval_steps=10),
        eval_every_n_steps=25,
        max_steps=99,
    )

    assert reporter.should_collect_model_scalars(phase="train", step=None, trainer=trainer) is False
    assert reporter.should_collect_model_scalars(phase="train", step=2, trainer=trainer) is False
    assert reporter.should_collect_model_scalars(phase="train", step=1, trainer=trainer) is True
    assert reporter.should_collect_model_scalars(phase="train", step=10, trainer=trainer) is True
    assert reporter.should_collect_model_scalars(phase="train", step=25, trainer=trainer) is True
    assert reporter.should_collect_model_scalars(phase="train", step=99, trainer=trainer) is True
