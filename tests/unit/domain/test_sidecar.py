from __future__ import annotations

import pytest

from taac2026.domain.config import (
    PCVRDataConfig,
    PCVRDataPipelineConfig,
    PCVRSequenceCropConfig,
    PCVRLossConfig,
    PCVRLossTermConfig,
    PCVRTrainConfig,
)
from taac2026.domain.sidecar import (
    build_pcvr_train_config_sidecar,
    load_pcvr_train_config_sidecar,
    validate_complete_config,
)


def test_load_pcvr_train_config_sidecar_round_trips_typed_config() -> None:
    config = PCVRTrainConfig(data=PCVRDataConfig(batch_size=32))

    loaded = load_pcvr_train_config_sidecar(
        build_pcvr_train_config_sidecar(config),
        config_type=PCVRTrainConfig,
    )

    assert loaded == config


def test_load_pcvr_train_config_sidecar_rejects_flat_payload() -> None:
    with pytest.raises(ValueError):
        load_pcvr_train_config_sidecar({"batch_size": 64}, config_type=PCVRTrainConfig)


def test_load_pcvr_train_config_sidecar_rejects_unknown_format() -> None:
    payload = build_pcvr_train_config_sidecar(PCVRTrainConfig())
    payload["train_config_format"] = "unknown-format"

    with pytest.raises(ValueError, match="unsupported PCVR train_config format"):
        load_pcvr_train_config_sidecar(payload, config_type=PCVRTrainConfig)


def test_validate_complete_config_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="incomplete train_config sidecar; missing fields:"):
        validate_complete_config({}, PCVRTrainConfig)


def test_validate_complete_config_rejects_missing_model_section() -> None:
    payload = PCVRTrainConfig().model_dump(mode="json")
    payload.pop("model")

    with pytest.raises(ValueError, match=r"missing fields: \['model'\]"):
        validate_complete_config(payload, PCVRTrainConfig)


def test_validate_complete_config_rejects_missing_optimizer_section() -> None:
    payload = PCVRTrainConfig().model_dump(mode="json")
    payload.pop("optimizer")

    with pytest.raises(ValueError, match=r"missing fields: \['optimizer'\]"):
        validate_complete_config(payload, PCVRTrainConfig)


def test_validate_complete_config_rejects_nested_missing_field() -> None:
    payload = PCVRTrainConfig().model_dump(mode="json")
    payload["model"].pop("d_model")

    with pytest.raises(ValueError, match=r"missing fields: \['model\.d_model'\]"):
        validate_complete_config(payload, PCVRTrainConfig)


def test_validate_complete_config_rejects_missing_loss_term_field() -> None:
    config = PCVRTrainConfig(
        loss=PCVRLossConfig(terms=(PCVRLossTermConfig(name="bce"),))
    )
    payload = config.model_dump(mode="json")
    payload["loss"]["terms"][0].pop("kind")

    with pytest.raises(ValueError, match=r"loss\.terms\[0\]\.kind"):
        validate_complete_config(payload, PCVRTrainConfig)


def test_validate_complete_config_rejects_missing_transform_field() -> None:
    config = PCVRTrainConfig(
        data_pipeline=PCVRDataPipelineConfig(
            transforms=(PCVRSequenceCropConfig(seq_window_min_len=4),)
        )
    )
    payload = config.model_dump(mode="json")
    payload["data_pipeline"]["transforms"][0].pop("views_per_row")

    with pytest.raises(ValueError, match=r"data_pipeline\.transforms\[0\]\.views_per_row"):
        validate_complete_config(payload, PCVRTrainConfig)


def test_validate_complete_config_accepts_sequence_entries_with_extra_items() -> None:
    config = PCVRTrainConfig(
        loss=PCVRLossConfig(
            terms=(
                PCVRLossTermConfig(name="bce"),
                PCVRLossTermConfig(name="focal", kind="focal", weight=0.5),
            )
        )
    )
    payload = config.model_dump(mode="json")
    # Dropping a whole term shortens the sequence; completeness only checks existing items.
    payload["loss"]["terms"] = payload["loss"]["terms"][:1]

    rebuilt = validate_complete_config(payload, PCVRTrainConfig)
    assert len(rebuilt.loss.terms) == 1


def test_validate_complete_config_rejects_unknown_fields() -> None:
    payload = PCVRTrainConfig().model_dump(mode="json")
    payload["unknown_section"] = {"x": 1}

    with pytest.raises(ValueError):
        validate_complete_config(payload, PCVRTrainConfig)


def test_validate_complete_config_accepts_full_train_config() -> None:
    config = PCVRTrainConfig(data=PCVRDataConfig(batch_size=16))
    rebuilt = validate_complete_config(config.model_dump(mode="json"), PCVRTrainConfig)
    assert rebuilt == config
