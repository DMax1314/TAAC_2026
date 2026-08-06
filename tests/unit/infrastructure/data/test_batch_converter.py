from __future__ import annotations

from pathlib import Path

import pytest
import pyarrow as pa

from taac2026.api import compute_sequence_stats
from taac2026.infrastructure.data.batch_converter import PCVRRecordBatchConverter, build_pcvr_column_plan
from taac2026.infrastructure.data.schema_layout import load_pcvr_schema_layout
from taac2026.infrastructure.io.json import write_path


def _write_schema(tmp_path: Path) -> Path:
    return write_path(
        tmp_path / "schema.json",
        {
            "format": "raw_parquet",
            "user_int": [[1, 10, 1], [2, 20, 2]],
            "item_int": [[3, 10, 1]],
            "user_dense": [[4, 2]],
            "seq": {
                "seq_a": {
                    "prefix": "domain_a_seq",
                    "ts_fid": 9,
                    "features": [[9, 128], [10, 16], [11, 16]],
                }
            },
        },
    )


def _record_batch() -> pa.RecordBatch:
    names = [
        "timestamp",
        "label_type",
        "user_id",
        "user_int_feats_1",
        "user_int_feats_2",
        "item_int_feats_3",
        "user_dense_feats_4",
        "domain_a_seq_9",
        "domain_a_seq_10",
        "domain_a_seq_11",
    ]
    arrays = [
        pa.array([100, 100], type=pa.int64()),
        pa.array([2, 1], type=pa.int64()),
        pa.array(["u0", "u1"]),
        pa.array([None, 5], type=pa.int64()),
        pa.array([[1], []], type=pa.list_(pa.int64())),
        pa.array([0, 7], type=pa.int64()),
        pa.array([[1.5], []], type=pa.list_(pa.float32())),
        pa.array([[10, 20, 30, 40], [90, 95]], type=pa.list_(pa.int64())),
        pa.array([[1, 2, 2, 3], [4, 4]], type=pa.list_(pa.int64())),
        pa.array([[1, 2, 2, 3], [5, 5]], type=pa.list_(pa.int64())),
    ]
    return pa.record_batch(arrays, names=names)


def test_record_batch_converter_emits_missing_masks_raw_sequences_and_timestamps(tmp_path: Path) -> None:
    layout = load_pcvr_schema_layout(_write_schema(tmp_path), {"seq_a": 4})
    column_plan = build_pcvr_column_plan(layout, _record_batch().schema.names)
    converter = PCVRRecordBatchConverter(
        layout=layout,
        column_plan=column_plan,
        batch_size=2,
        clip_vocab=True,
        is_training=True,
        strict_time_filter=True,
    )

    batch = converter.convert(_record_batch())

    assert batch.inputs.user.int_missing_mask.tolist() == [[True, False, True], [False, True, True]]
    assert batch.inputs.item.int_missing_mask.tolist() == [[True], [False]]
    assert batch.inputs.user.dense_missing_mask.tolist() == [[False, True], [True, True]]
    assert batch.inputs.item.dense_missing_mask.shape == (2, 0)

    # Sequences stay raw (pre-dedup): duplicate events are kept for model-side
    # stats derivation; models deduplicate after computing sequence statistics.
    seq_a = batch.inputs.sequences["seq_a"]
    assert seq_a.lengths.tolist() == [4, 2]
    assert seq_a.values[0].tolist() == [[1, 2, 2, 3], [1, 2, 2, 3]]
    assert seq_a.values[1].tolist() == [[4, 4, 0, 0], [5, 5, 0, 0]]

    # Raw event timestamps are carried on the sequence input; time features are model-side.
    assert seq_a.timestamps.shape == (2, 4)
    assert seq_a.timestamps[0].tolist() == [10, 20, 30, 40]
    assert seq_a.timestamps[1].tolist() == [90, 95, 0, 0]
    assert batch.inputs.request_timestamp.tolist() == [100, 100]
    # label is derived from label_type == 2 during training conversion
    assert batch.label.tolist() == [1, 0]


def test_record_batch_converter_stats_derive_from_raw_pre_dedup_sequences(tmp_path: Path) -> None:
    """Converter output must feed model-side stats with duplicate events intact."""
    layout = load_pcvr_schema_layout(_write_schema(tmp_path), {"seq_a": 4})
    column_plan = build_pcvr_column_plan(layout, _record_batch().schema.names)
    converter = PCVRRecordBatchConverter(
        layout=layout,
        column_plan=column_plan,
        batch_size=2,
        clip_vocab=True,
        is_training=True,
        strict_time_filter=True,
    )

    batch = converter.convert(_record_batch())
    seq_a = batch.inputs.sequences["seq_a"]
    stats = compute_sequence_stats(
        seq_a.values,
        seq_a.lengths,
        seq_a.timestamps,
        batch.inputs.request_timestamp,
    )

    # [length, active_events, unique_events, dup_ratio, nonzero_fraction, last_gap_bucket]
    assert stats[:, 0].tolist() == [4.0, 2.0]
    assert stats[:, 1].tolist() == [4.0, 2.0]
    assert stats[:, 2].tolist() == [3.0, 1.0]
    assert stats[:, 3].tolist() == [pytest.approx(0.25), pytest.approx(0.5)]


def test_build_pcvr_column_plan_fails_when_item_dense_column_missing(tmp_path: Path) -> None:
    schema_path = write_path(
        tmp_path / "schema.json",
        {
            "format": "raw_parquet",
            "user_int": [[1, 10, 1]],
            "item_int": [[2, 10, 1]],
            "user_dense": [],
            "item_dense": [[5, 2]],
            "seq": {},
        },
    )
    layout = load_pcvr_schema_layout(schema_path, {})
    parquet_names = [
        "timestamp",
        "label_type",
        "user_id",
        "user_int_feats_1",
        "item_int_feats_2",
    ]

    with pytest.raises(KeyError, match="item_dense_feats_5"):
        build_pcvr_column_plan(layout, parquet_names)