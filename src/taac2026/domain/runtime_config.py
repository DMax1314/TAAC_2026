"""Runtime-facing PCVR configuration boundary objects."""

from __future__ import annotations

import math
from typing import Any

from pydantic import ConfigDict, field_validator, model_validator

from taac2026.domain.validation import TAACBoundaryModel


AMP_DTYPE_CHOICES: tuple[str, ...] = ("bfloat16", "float16")
PCVR_LOSS_TERM_KIND_CHOICES: tuple[str, ...] = ("bce", "focal", "pairwise_auc", "model")
DENSE_OPTIMIZER_TYPE_CHOICES: tuple[str, ...] = (
    "adamw",
    "fused_adamw",
    "orthogonal_adamw",
    "muon",
)
DEFAULT_PROGRESS_LOG_INTERVAL_STEPS = 100
_AMP_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
}


def normalize_amp_dtype(value: str | None) -> str:
    if value is None:
        return "bfloat16"
    normalized = str(value).strip().lower()
    try:
        return _AMP_DTYPE_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported amp dtype: {value}") from error


class RuntimeExecutionConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    amp: bool = False
    amp_dtype: str = "bfloat16"
    compile: bool = False
    progress_log_interval_steps: int = DEFAULT_PROGRESS_LOG_INTERVAL_STEPS
    deterministic: bool = True

    @field_validator("progress_log_interval_steps")
    @classmethod
    def _validate_progress_log_interval_steps(cls, value: int) -> int:
        interval = int(value)
        if interval <= 0:
            raise ValueError("progress_log_interval_steps must be positive")
        return interval

    @field_validator("amp_dtype")
    @classmethod
    def _normalize_amp_dtype(cls, value: str) -> str:
        return normalize_amp_dtype(value)

    @field_validator("deterministic")
    @classmethod
    def _coerce_deterministic(cls, value: Any) -> bool:
        return bool(value)

    def normalized_amp_dtype(self) -> str:
        return normalize_amp_dtype(self.amp_dtype)


class PCVRLossTermConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: str = "bce"
    weight: float = 1.0
    focal_alpha: float = 0.1
    focal_gamma: float = 2.0
    temperature: float = 1.0

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("loss term name must be non-empty")
        return name

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        kind = str(value).strip().lower()
        if kind not in PCVR_LOSS_TERM_KIND_CHOICES:
            raise ValueError(f"unsupported PCVR loss term kind: {value}")
        return kind

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"loss term weight must be finite and >= 0, got {value}")
        return weight

    @field_validator("focal_alpha")
    @classmethod
    def _validate_focal_alpha(cls, value: float) -> float:
        focal_alpha = float(value)
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError(f"focal_alpha must be between 0 and 1, got {value}")
        return focal_alpha

    @field_validator("focal_gamma")
    @classmethod
    def _validate_focal_gamma(cls, value: float) -> float:
        focal_gamma = float(value)
        if focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0, got {value}")
        return focal_gamma

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        temperature = float(value)
        if temperature <= 0.0:
            raise ValueError(f"loss term temperature must be > 0, got {value}")
        return temperature

def _default_pcvr_loss_terms() -> tuple[PCVRLossTermConfig, ...]:
    return (PCVRLossTermConfig(name="bce", kind="bce", weight=1.0),)


class PCVRLossConfig(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    terms: tuple[PCVRLossTermConfig, ...] = _default_pcvr_loss_terms()

    @model_validator(mode="after")
    def _validate_terms(self) -> PCVRLossConfig:
        if not self.terms:
            raise ValueError("PCVR loss config must define at least one loss term")
        names = [term.name for term in self.terms]
        duplicate_names = sorted({name for name in names if names.count(name) > 1})
        if duplicate_names:
            joined = ", ".join(duplicate_names)
            raise ValueError(f"PCVR loss term names must be unique; duplicates: {joined}")
        return self

    def summary(self) -> str:
        return ", ".join(f"{term.name}:{term.kind}*{term.weight:g}" for term in self.terms)


DEFAULT_PCVR_LOSS_CONFIG = PCVRLossConfig()


__all__ = [
    "AMP_DTYPE_CHOICES",
    "DEFAULT_PCVR_LOSS_CONFIG",
    "DEFAULT_PROGRESS_LOG_INTERVAL_STEPS",
    "DENSE_OPTIMIZER_TYPE_CHOICES",
    "PCVR_LOSS_TERM_KIND_CHOICES",
    "PCVRLossConfig",
    "PCVRLossTermConfig",
    "RuntimeExecutionConfig",
    "normalize_amp_dtype",
]
