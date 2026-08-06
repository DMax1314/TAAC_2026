"""Factory helpers for PCVR experiment packages."""

from __future__ import annotations

from pathlib import Path

import torch

from taac2026.domain.config import PCVRTrainConfig
from taac2026.application.experiments.experiment import PCVRExperiment


def create_pcvr_experiment(
    *,
    name: str,
    package_dir: Path,
    model_type: type[torch.nn.Module],
    train_defaults: PCVRTrainConfig,
    config_type: type[PCVRTrainConfig] = PCVRTrainConfig,
) -> PCVRExperiment:
    """Create a PCVR experiment from a typed declaration."""

    return PCVRExperiment(
        name=name,
        package_dir=package_dir,
        model_type=model_type,
        config_type=config_type,
        train_defaults=train_defaults,
    )


__all__ = ["create_pcvr_experiment"]
