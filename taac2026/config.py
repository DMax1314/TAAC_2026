from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    dataset_path: str
    max_seq_len: int = 96
    max_feature_tokens: int = 64
    max_event_features: int = 12
    val_ratio: float = 0.2
    label_action_type: int = 2


@dataclass(slots=True)
class ModelConfig:
    name: str = "baseline"
    vocab_size: int = 200_003
    embedding_dim: int = 96
    hidden_dim: int = 192
    dropout: float = 0.15
    num_layers: int = 2
    num_heads: int = 4
    recent_seq_len: int = 32
    memory_slots: int = 12
    ffn_multiplier: int = 4


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"
    output_dir: str = "outputs/baseline"


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}

    defaults = {
        "data": asdict(DataConfig(dataset_path="")),
        "model": asdict(ModelConfig()),
        "train": asdict(TrainConfig()),
    }
    merged = _merge_dict(defaults, raw)

    return ExperimentConfig(
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
    )
