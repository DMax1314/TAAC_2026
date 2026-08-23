from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from taac2026.domain.config import PCVRNSConfig
from taac2026.domain.schema import PCVRSchema
from taac2026.infrastructure.modeling.model_contract import (
    build_feature_specs,
    build_pcvr_model_specs,
    dataset_schema_path,
    load_ns_groups,
    parse_seq_max_lens,
    resolve_checkpoint_schema_path,
    resolve_schema_path,
    resolve_training_schema_path,
)
from taac2026.infrastructure.modeling.time_features import (
    NUM_TIME_BUCKETS,
    SEQUENCE_STATS_DIM,
    compute_sequence_stats,
    compute_sequence_time_buckets,
)


def _make_schema(user_fids: list[int] = (10, 20), item_fids: list[int] = (7,)) -> PCVRSchema:
    return PCVRSchema(
        format="raw_parquet",
        user_int=tuple([fid, 10, 1] for fid in user_fids),
        item_int=tuple([fid, 10, 1] for fid in item_fids),
        user_dense=[[4, 4]],
        seq={},
    )


def test_parse_seq_max_lens_and_feature_specs() -> None:
    schema = SimpleNamespace(entries=[(10, 0, 1), (20, 1, 2)])

    assert parse_seq_max_lens("seq_a:4, seq_b:8") == {"seq_a": 4, "seq_b": 8}
    assert build_feature_specs(schema, [11, 21, 22]) == [(11, 0, 1), (22, 1, 2)]


def test_resolve_schema_path_prefers_explicit_path(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    checkpoint_dir = tmp_path / "ckpt" / "global_step1"
    explicit_schema = tmp_path / "explicit_schema.json"
    dataset_dir.mkdir()
    checkpoint_dir.mkdir(parents=True)
    explicit_schema.write_text("{}", encoding="utf-8")
    (dataset_dir / "schema.json").write_text("{}", encoding="utf-8")

    assert resolve_schema_path(explicit_schema, fallback=checkpoint_dir / "schema.json") == explicit_schema.resolve()


def test_resolve_schema_path_fails_fast_on_missing_explicit_path(tmp_path) -> None:
    missing_schema = tmp_path / "missing_schema.json"
    checkpoint_dir = tmp_path / "ckpt" / "global_step1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "schema.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="explicit path"):
        resolve_schema_path(missing_schema, fallback=checkpoint_dir / "schema.json")


def test_resolve_training_schema_path_uses_dataset_fallback(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    dataset_schema = dataset_dir / "schema.json"
    dataset_schema.write_text("{}", encoding="utf-8")

    assert resolve_training_schema_path(dataset_dir, None) == dataset_schema.resolve()

    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="fallback path"):
        resolve_training_schema_path(empty_dir, None)


def test_resolve_training_schema_path_fails_on_missing_explicit(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    (dataset_dir / "schema.json").write_text("{}", encoding="utf-8")
    missing = tmp_path / "explicit_schema.json"

    with pytest.raises(FileNotFoundError, match="explicit path"):
        resolve_training_schema_path(dataset_dir, missing)


def test_resolve_checkpoint_schema_path_uses_checkpoint_fallback(tmp_path) -> None:
    checkpoint_dir = tmp_path / "ckpt" / "global_step1"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_schema = checkpoint_dir / "schema.json"
    checkpoint_schema.write_text("{}", encoding="utf-8")

    assert resolve_checkpoint_schema_path(checkpoint_dir, None) == checkpoint_schema.resolve()

    with pytest.raises(FileNotFoundError, match="fallback path"):
        resolve_checkpoint_schema_path(tmp_path / "empty", None)


def test_resolve_checkpoint_schema_path_fails_on_missing_explicit(tmp_path) -> None:
    checkpoint_dir = tmp_path / "ckpt" / "global_step1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "schema.json").write_text("{}", encoding="utf-8")
    missing = tmp_path / "explicit_schema.json"

    with pytest.raises(FileNotFoundError, match="explicit path"):
        resolve_checkpoint_schema_path(checkpoint_dir, missing)


def test_dataset_schema_path_uses_dataset_directory_or_file_parent(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "train.parquet"
    dataset_file.touch()

    expected = (dataset_dir / "schema.json").resolve()
    assert dataset_schema_path(dataset_dir) == expected
    assert dataset_schema_path(dataset_file) == expected


def test_load_ns_groups_maps_feature_ids_to_schema_positions() -> None:
    schema = _make_schema(user_fids=[10, 20], item_fids=[7])

    assert load_ns_groups(
        schema,
        PCVRNSConfig(
            grouping_strategy="explicit",
            user_groups={"u": [20, 10]},
            item_groups={"i": [7]},
        ),
    ) == ([[1, 0]], [[0]])


def test_load_ns_groups_supports_explicit_singleton_strategy() -> None:
    schema = _make_schema(user_fids=[10, 20], item_fids=[7])

    assert load_ns_groups(
        schema,
        PCVRNSConfig(grouping_strategy="singleton", user_groups={}, item_groups={}),
    ) == ([[0], [1]], [[0]])


def test_load_ns_groups_explicit_empty_groups_yield_empty_lists() -> None:
    schema = _make_schema(user_fids=[10], item_fids=[7])

    assert load_ns_groups(
        schema,
        PCVRNSConfig(grouping_strategy="explicit", user_groups={}, item_groups={}),
    ) == ([], [])


def test_load_ns_groups_rejects_unknown_feature_id() -> None:
    schema = _make_schema(user_fids=[10], item_fids=[7])

    with pytest.raises(KeyError, match="999"):
        load_ns_groups(
            schema,
            PCVRNSConfig(
                grouping_strategy="explicit",
                user_groups={"u": [999]},
                item_groups={"i": [7]},
            ),
        )


def test_build_pcvr_model_specs_compiles_schema_derived_inputs() -> None:
    schema = _make_schema(user_fids=[10, 20], item_fids=[7])

    specs = build_pcvr_model_specs(
        schema,
        PCVRNSConfig(
            grouping_strategy="explicit",
            user_groups={"u": [20, 10]},
            item_groups={"i": [7]},
        ),
    )

    assert specs.user_int_feature_specs == [(10, 0, 1), (10, 1, 1)]
    assert specs.item_int_feature_specs == [(10, 0, 1)]
    assert specs.user_dense_dim == 4
    assert specs.item_dense_dim == 0
    assert specs.seq_vocab_sizes == {}
    assert specs.user_ns_groups == [[1, 0]]
    assert specs.item_ns_groups == [[0]]


def test_compute_sequence_time_buckets_zero_timestamps_map_to_padding_bucket() -> None:
    seq_timestamps = torch.tensor([[0, 100, 1000, 86400]], dtype=torch.int64)
    request_timestamps = torch.tensor([86400 * 10], dtype=torch.int64)

    buckets = compute_sequence_time_buckets(seq_timestamps, request_timestamps)

    assert buckets.shape == (1, 4)
    assert buckets[0, 0].item() == 0
    assert buckets.max().item() < NUM_TIME_BUCKETS
    assert torch.any(buckets > 0)


def test_compute_sequence_time_buckets_is_batch_wise() -> None:
    seq_timestamps = torch.tensor([[10], [86400 * 365]], dtype=torch.int64)
    request_timestamps = torch.tensor([86400, 86400 * 365], dtype=torch.int64)

    buckets = compute_sequence_time_buckets(seq_timestamps, request_timestamps)

    # Older events (larger request-relative gap) map to larger bucket ids.
    assert buckets[0, 0] > buckets[1, 0]


def test_compute_sequence_stats_shape_and_semantics() -> None:
    sequence = torch.tensor([[[3, 3, 0], [1, 1, 0]]], dtype=torch.int64)
    lengths = torch.tensor([3], dtype=torch.int64)
    seq_timestamps = torch.tensor([[10, 20, 0]], dtype=torch.int64)
    request_timestamps = torch.tensor([100], dtype=torch.int64)

    stats = compute_sequence_stats(sequence, lengths, seq_timestamps, request_timestamps)

    assert stats.shape == (1, SEQUENCE_STATS_DIM)
    assert stats[0, 0].item() == 3  # length
    assert stats[0, 1].item() == 2  # active events (position 2 is zero-padded)
    assert stats[0, 2].item() == 1  # unique rows: (3, 1) appears twice
    assert stats[0, 3].item() == pytest.approx(0.5)  # 1 - unique/active
    assert stats[0, 4].item() == pytest.approx(4 / 6)  # nonzero fraction
    assert stats[0, 5].item() == 0  # last event is padded, bucket 0
