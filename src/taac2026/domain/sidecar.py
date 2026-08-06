"""PCVR train_config.json sidecar helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import field_validator

from taac2026.domain.config import PCVRTrainConfig
from taac2026.domain.validation import TAACBoundaryModel


PCVR_TRAIN_CONFIG_FORMAT = "taac2026-pcvr-train-config"

PCVR_TRAIN_CONFIG_METADATA_KEYS = frozenset(
    {
        "train_config_format",
        "framework_name",
    }
)


class PCVRTrainConfigSidecar(TAACBoundaryModel):
    """Pydantic model for the PCVR train_config payload."""

    train_config_format: str
    framework_name: str
    train_config: dict[str, Any]

    @field_validator("train_config_format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        if value != PCVR_TRAIN_CONFIG_FORMAT:
            raise ValueError(f"unsupported PCVR train_config format: {value}")
        return value

    @field_validator("framework_name")
    @classmethod
    def _validate_framework_name(cls, value: str) -> str:
        if value != "taac2026":
            raise ValueError(f"unsupported framework_name: {value}")
        return value


def build_pcvr_train_config_sidecar(train_config: PCVRTrainConfig) -> dict[str, Any]:
    """Serialize a typed train config into the PCVR train_config.json payload."""

    return PCVRTrainConfigSidecar.model_validate(
        {
            "train_config_format": PCVR_TRAIN_CONFIG_FORMAT,
            "framework_name": "taac2026",
            "train_config": train_config.model_dump(mode="json"),
        }
    ).model_dump(mode="python")


def _missing_keys(
    payload: Any,
    expected: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Recursively find keys present in ``expected`` but missing from ``payload``.

    Recurses into both mappings and sequences so nested config entries (loss
    terms, data pipeline transforms) are validated as complete snapshots too.
    """
    missing: list[str] = []
    if isinstance(expected, Mapping) and isinstance(payload, Mapping):
        for key, value in expected.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if key not in payload:
                missing.append(full_key)
                continue
            missing.extend(_missing_keys(payload[key], value, prefix=full_key))
    elif isinstance(expected, list) and isinstance(payload, list):
        for index, (expected_item, actual_item) in enumerate(zip(expected, payload, strict=False)):
            missing.extend(_missing_keys(actual_item, expected_item, prefix=f"{prefix}[{index}]"))
    return missing


def validate_complete_config(
    payload: Mapping[str, Any],
    config_type: type[PCVRTrainConfig],
) -> PCVRTrainConfig:
    """Validate a full config snapshot; missing fields fail instead of defaulting."""
    config = config_type.model_validate(dict(payload))
    missing = _missing_keys(payload, config.model_dump(mode="json"))
    if missing:
        raise ValueError(f"incomplete train_config sidecar; missing fields: {missing}")
    return config


def load_pcvr_train_config_sidecar(
    payload: Mapping[str, Any],
    *,
    config_type: type[PCVRTrainConfig],
) -> PCVRTrainConfig:
    """Load and validate the structured payload back into a typed train config."""

    config_body = PCVRTrainConfigSidecar.model_validate(dict(payload)).train_config
    return validate_complete_config(config_body, config_type)


__all__ = [
    "PCVR_TRAIN_CONFIG_FORMAT",
    "PCVR_TRAIN_CONFIG_METADATA_KEYS",
    "PCVRTrainConfigSidecar",
    "build_pcvr_train_config_sidecar",
    "load_pcvr_train_config_sidecar",
    "validate_complete_config",
]
