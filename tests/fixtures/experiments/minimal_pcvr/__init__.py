from pathlib import Path

from taac2026.api import PCVRTrainConfig, create_pcvr_experiment

from .model import PCVRMinimalExp


TRAIN_DEFAULTS = PCVRTrainConfig()

EXPERIMENT = create_pcvr_experiment(
    name="pcvr_minimal_exp",
    package_dir=Path(__file__).resolve().parent,
    model_type=PCVRMinimalExp,
    train_defaults=TRAIN_DEFAULTS,
)

__all__ = ["EXPERIMENT", "TRAIN_DEFAULTS"]
