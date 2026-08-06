"""Contract tests for the validated, immutable ``PCVRSchema`` payload."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from taac2026.domain.schema import PCVRSchema


def _valid_schema_payload() -> dict[str, object]:
    return {
        "format": "raw_parquet",
        "user_int": [[1, 10, 1], [2, 20, 2]],
        "item_int": [[3, 10, 1]],
        "user_dense": [[4, 2]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 1000], [11, 100]],
            }
        },
    }


def test_accepts_list_form_and_exposes_normalized_columns() -> None:
    schema = PCVRSchema.model_validate(_valid_schema_payload())
    assert schema.user_int[0].fid == 1
    assert schema.user_int[0].vocab_size == 10
    assert schema.user_int[0].dim == 1
    assert schema.item_dense == ()
    assert schema.seq["seq_a"].ts_fid == 10
    assert schema.seq["seq_a"].features[1].fid == 11


def test_accepts_dict_form_columns() -> None:
    payload = _valid_schema_payload()
    payload["user_int"] = [{"fid": 1, "vocab_size": 10, "dim": 1}]
    payload["item_dense"] = [{"fid": 5, "dim": 3}]
    schema = PCVRSchema.model_validate(payload)
    assert schema.user_int[0].fid == 1
    assert schema.item_dense[0].dim == 3


def test_rejects_unknown_fields() -> None:
    payload = _valid_schema_payload()
    payload["pair"] = {"user": [1, 2]}
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_unknown_sequence_field() -> None:
    payload = _valid_schema_payload()
    payload["seq"]["seq_a"]["global_time"] = True  # type: ignore[typeddict-item]
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_is_immutable() -> None:
    schema = PCVRSchema.model_validate(_valid_schema_payload())
    with pytest.raises(ValidationError):
        schema.user_int = ()  # type: ignore[misc]


def test_rejects_empty_int_groups() -> None:
    payload = _valid_schema_payload()
    payload["user_int"] = []
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_duplicate_fid_in_group() -> None:
    payload = _valid_schema_payload()
    payload["item_int"] = [[3, 10, 1], [3, 20, 1]]
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_invalid_dim_and_vocab() -> None:
    payload = _valid_schema_payload()
    payload["user_dense"] = [[4, 0]]
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)
    payload = _valid_schema_payload()
    payload["user_int"] = [[1, -1, 1]]
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_malformed_column_lists() -> None:
    payload = _valid_schema_payload()
    payload["user_int"] = [[1, 10]]
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_ts_fid_not_in_features() -> None:
    payload = _valid_schema_payload()
    payload["seq"]["seq_a"]["ts_fid"] = 99
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_duplicate_sequence_prefix() -> None:
    payload = _valid_schema_payload()
    payload["seq"]["seq_b"] = {
        "prefix": "domain_a_seq",
        "ts_fid": 20,
        "features": [[20, 100], [21, 20]],
    }
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_empty_sequence_features() -> None:
    payload = _valid_schema_payload()
    payload["seq"]["seq_a"]["features"] = []
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)


def test_rejects_non_raw_parquet_format() -> None:
    payload = _valid_schema_payload()
    payload["format"] = "pair_views"
    with pytest.raises(ValidationError):
        PCVRSchema.model_validate(payload)
