from __future__ import annotations

import pytest

from taac2026.domain.sidecar import (
    PCVR_TRAIN_CONFIG_FORMAT,
    build_pcvr_train_config_sidecar,
    load_pcvr_train_config_sidecar,
)


def test_load_pcvr_train_config_sidecar_accepts_current_payload() -> None:
    payload = build_pcvr_train_config_sidecar({"batch_size": 32})

    loaded = load_pcvr_train_config_sidecar(payload)

    assert loaded["batch_size"] == 32
    assert loaded["train_config_format"] == PCVR_TRAIN_CONFIG_FORMAT


def test_load_pcvr_train_config_sidecar_rejects_flat_payload() -> None:
    with pytest.raises(ValueError):
        load_pcvr_train_config_sidecar({"batch_size": 64})


def test_load_pcvr_train_config_sidecar_rejects_unknown_format() -> None:
    payload = build_pcvr_train_config_sidecar({"batch_size": 32})
    payload["train_config_format"] = "unknown-format"

    with pytest.raises(ValueError, match="unsupported PCVR train_config format"):
        load_pcvr_train_config_sidecar(payload)
