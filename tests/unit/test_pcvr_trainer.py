from __future__ import annotations

import logging

import torch

import taac2026.infrastructure.pcvr.trainer as trainer_module
from taac2026.infrastructure.pcvr.trainer import PCVRPointwiseTrainer
from taac2026.infrastructure.training.runtime import EarlyStopping


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, model_input):
        del model_input
        return self.bias.view(1, 1)

    def predict(self, model_input):
        logits = self.forward(model_input)
        return logits, torch.empty(0)


def test_train_logs_progress_when_tqdm_is_disabled(monkeypatch, tmp_path, caplog) -> None:
    train_loader = [{"label": torch.tensor([0.0])} for _ in range(4)]
    valid_loader = [{"label": torch.tensor([0.0])} for _ in range(3)]
    trainer = PCVRPointwiseTrainer(
        model=_DummyModel(),
        model_input_type=object,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        save_dir=tmp_path / "checkpoints",
        early_stopping=EarlyStopping(tmp_path / "best" / "model.pt", patience=2),
    )

    losses = iter((0.5, 0.4, 0.3, 0.2))
    monotonic_values = iter((100.0, 101.0, 102.0, 103.0, 104.0))
    monkeypatch.setattr(trainer_module, "_use_interactive_progress", lambda: False)
    monkeypatch.setattr(trainer_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(trainer, "_train_step", lambda batch: next(losses))
    monkeypatch.setattr(trainer, "evaluate", lambda epoch=None: (0.75, 0.25))

    with caplog.at_level(logging.INFO):
        trainer.train()

    messages = [record.getMessage() for record in caplog.records]
    assert "Train epoch 1 progress 1/4 (25.0%) | eta=0:00:03 | loss=0.5000" in messages
    assert "Train epoch 1 progress 4/4 (100.0%) | eta=0:00:00 | loss=0.2000" in messages
    assert "Epoch 1, Average Loss: 0.35" in messages