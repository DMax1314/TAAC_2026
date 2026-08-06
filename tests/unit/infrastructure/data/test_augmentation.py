from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from taac2026.infrastructure.io.json import dumps
from taac2026.domain.config import (
    PCVRDataCacheConfig,
    PCVRDataPipelineConfig,
    PCVRDomainDropoutConfig,
    PCVRFeatureMaskConfig,
    PCVRNonSequentialSparseDropoutConfig,
    PCVRSequenceCropConfig,
)
from taac2026.infrastructure.data import cache as cache_module
from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVREntityInput,
    PCVRModelInput,
    PCVRSequenceInput,
)
from taac2026.infrastructure.data.dataset import PCVRParquetDataset
from taac2026.infrastructure.data.pipeline import (
    PCVRDataPipeline,
    PCVRDomainDropoutTransform,
    PCVRFeatureMaskTransform,
    PCVRMemoryBatchCache,
    PCVRNonSequentialSparseDropoutTransform,
    PCVRSharedBatchCache,
    PCVRSharedTensorSpec,
    PCVRSequenceCropTransform,
    build_pcvr_batch_transforms,
    concat_pcvr_batches,
)


def _make_batch() -> PCVRBatch:
    user = PCVREntityInput(
        int_values=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        int_missing_mask=torch.zeros(2, 2, dtype=torch.bool),
        dense_values=torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        dense_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
    )
    item = PCVREntityInput(
        int_values=torch.tensor([[5], [6]], dtype=torch.long),
        int_missing_mask=torch.zeros(2, 1, dtype=torch.bool),
        dense_values=torch.zeros(2, 0, dtype=torch.float32),
        dense_missing_mask=torch.zeros(2, 0, dtype=torch.bool),
    )
    seq_a = PCVRSequenceInput(
        values=torch.tensor(
            [
                [[1, 2, 3, 4]],
                [[7, 8, 9, 0]],
            ],
            dtype=torch.long,
        ),
        lengths=torch.tensor([4, 3], dtype=torch.long),
        timestamps=torch.tensor(
            [
                [40, 30, 20, 10],
                [30, 20, 10, 0],
            ],
            dtype=torch.long,
        ),
    )
    inputs = PCVRModelInput(
        user=user,
        item=item,
        sequences={"seq_a": seq_a},
        request_timestamp=torch.tensor([100, 200], dtype=torch.long),
    )
    return PCVRBatch(
        inputs=inputs,
        label=torch.tensor([1, 0], dtype=torch.long),
        user_id=["u0", "u1"],
    )


def _shared_label_specs(batch_size: int) -> dict[str, PCVRSharedTensorSpec]:
    """Shared cache specs with empty entities and sequences, plus label and timestamp."""
    return {
        "user_int_values": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.long),
        "user_int_missing_mask": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.bool),
        "user_dense_values": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.float32),
        "user_dense_missing_mask": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.bool),
        "item_int_values": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.long),
        "item_int_missing_mask": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.bool),
        "item_dense_values": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.float32),
        "item_dense_missing_mask": PCVRSharedTensorSpec(shape=(batch_size, 0), dtype=torch.bool),
        "label": PCVRSharedTensorSpec(shape=(batch_size,), dtype=torch.long),
        "request_timestamp": PCVRSharedTensorSpec(shape=(batch_size,), dtype=torch.long),
    }


def _label_only_batch(labels: list[int]) -> PCVRBatch:
    batch_size = len(labels)
    empty = PCVREntityInput(
        int_values=torch.zeros(batch_size, 0, dtype=torch.long),
        int_missing_mask=torch.zeros(batch_size, 0, dtype=torch.bool),
        dense_values=torch.zeros(batch_size, 0, dtype=torch.float32),
        dense_missing_mask=torch.zeros(batch_size, 0, dtype=torch.bool),
    )
    inputs = PCVRModelInput(
        user=empty,
        item=empty,
        sequences={},
        request_timestamp=torch.zeros(batch_size, dtype=torch.long),
    )
    return PCVRBatch(
        inputs=inputs,
        label=torch.tensor(labels, dtype=torch.long),
        user_id=[],
    )


def _label_only_batch_with_user_ids(labels: list[int], user_ids: list[str]) -> PCVRBatch:
    batch = _label_only_batch(labels)
    return PCVRBatch(inputs=batch.inputs, label=batch.label, user_id=user_ids)


def _assert_batch_equal(left: PCVRBatch, right: PCVRBatch) -> None:
    assert left.label.shape == right.label.shape
    assert left.user_id == right.user_id
    assert torch.equal(left.label, right.label)
    assert torch.equal(left.inputs.request_timestamp, right.inputs.request_timestamp)
    for entity_name in ("user", "item"):
        left_entity = getattr(left.inputs, entity_name)
        right_entity = getattr(right.inputs, entity_name)
        assert torch.equal(left_entity.int_values, right_entity.int_values)
        assert torch.equal(left_entity.int_missing_mask, right_entity.int_missing_mask)
        assert torch.equal(left_entity.dense_values, right_entity.dense_values)
        assert torch.equal(left_entity.dense_missing_mask, right_entity.dense_missing_mask)
    assert left.inputs.sequences.keys() == right.inputs.sequences.keys()
    for domain, left_sequence in left.inputs.sequences.items():
        right_sequence = right.inputs.sequences[domain]
        assert torch.equal(left_sequence.values, right_sequence.values)
        assert torch.equal(left_sequence.lengths, right_sequence.lengths)
        assert torch.equal(left_sequence.timestamps, right_sequence.timestamps)


def test_empty_pipeline_config_builds_no_transforms() -> None:
    config = PCVRDataPipelineConfig()

    assert config.transform_names == ()
    assert build_pcvr_batch_transforms(config) == ()


def test_disabled_transform_preserves_batch_content() -> None:
    batch = _make_batch()
    transform = PCVRSequenceCropTransform(PCVRSequenceCropConfig(enabled=False))

    augmented = transform(batch, generator=torch.Generator().manual_seed(1))

    _assert_batch_equal(augmented, batch)
    assert augmented is not batch


def test_sequence_crop_expands_rows_and_keeps_metadata_aligned() -> None:
    batch = _make_batch()
    transform = PCVRSequenceCropTransform(
        PCVRSequenceCropConfig(
            views_per_row=2,
            seq_window_mode="random_tail",
            seq_window_min_len=2,
        )
    )

    augmented = transform(batch, generator=torch.Generator().manual_seed(7))

    assert augmented.label.tolist() == [1, 1, 0, 0]
    assert augmented.inputs.request_timestamp.tolist() == [100, 100, 200, 200]
    assert augmented.user_id == ["u0", "u0", "u1", "u1"]
    assert augmented.inputs.user.int_values.shape == (4, 2)
    seq_a = augmented.inputs.sequences["seq_a"]
    assert seq_a.values.shape == (4, 1, 4)
    assert seq_a.lengths.min().item() >= 2
    assert seq_a.lengths.max().item() <= 4
    for row_index, length in enumerate(seq_a.lengths.tolist()):
        assert torch.equal(
            seq_a.values[row_index, :, length:],
            torch.zeros(1, 4 - length, dtype=torch.long),
        )
        assert torch.equal(
            seq_a.timestamps[row_index, length:],
            torch.zeros(4 - length, dtype=torch.long),
        )


def test_domain_dropout_clears_sequence_tokens_lengths_and_timestamps() -> None:
    batch = _make_batch()
    transform = PCVRDomainDropoutTransform(PCVRDomainDropoutConfig(probability=1.0))

    augmented = transform(batch, generator=torch.Generator().manual_seed(3))

    seq_a = augmented.inputs.sequences["seq_a"]
    assert torch.equal(seq_a.values, torch.zeros_like(seq_a.values))
    assert torch.equal(seq_a.lengths, torch.zeros_like(seq_a.lengths))
    assert torch.equal(seq_a.timestamps, torch.zeros_like(seq_a.timestamps))


def test_feature_masking_compacts_sequence_lengths() -> None:
    batch = _make_batch()
    transform = PCVRFeatureMaskTransform(PCVRFeatureMaskConfig(probability=1.0))

    augmented = transform(batch, generator=torch.Generator().manual_seed(5))

    assert torch.equal(
        augmented.inputs.user.int_values, torch.zeros_like(augmented.inputs.user.int_values)
    )
    assert torch.equal(
        augmented.inputs.item.int_values, torch.zeros_like(augmented.inputs.item.int_values)
    )
    seq_a = augmented.inputs.sequences["seq_a"]
    assert torch.equal(seq_a.values, torch.zeros_like(seq_a.values))
    assert torch.equal(seq_a.lengths, torch.zeros_like(seq_a.lengths))
    assert torch.equal(seq_a.timestamps, torch.zeros_like(seq_a.timestamps))
    assert torch.equal(
        augmented.inputs.user.int_missing_mask,
        torch.ones_like(augmented.inputs.user.int_missing_mask),
    )
    assert torch.equal(
        augmented.inputs.item.int_missing_mask,
        torch.ones_like(augmented.inputs.item.int_missing_mask),
    )


def test_nonseq_sparse_dropout_masks_full_rows_without_touching_sequences() -> None:
    batch = _make_batch()
    original_sequence = batch.inputs.sequences["seq_a"].values.clone()
    original_lengths = batch.inputs.sequences["seq_a"].lengths.clone()
    original_dense = batch.inputs.user.dense_values.clone()
    transform = PCVRNonSequentialSparseDropoutTransform(PCVRNonSequentialSparseDropoutConfig(probability=1.0))

    augmented = transform(batch, generator=torch.Generator().manual_seed(11))

    assert torch.equal(augmented.inputs.user.int_values, torch.zeros_like(augmented.inputs.user.int_values))
    assert torch.equal(augmented.inputs.item.int_values, torch.zeros_like(augmented.inputs.item.int_values))
    assert torch.equal(augmented.inputs.user.int_missing_mask, torch.ones_like(augmented.inputs.user.int_missing_mask))
    assert torch.equal(augmented.inputs.item.int_missing_mask, torch.ones_like(augmented.inputs.item.int_missing_mask))
    seq_a = augmented.inputs.sequences["seq_a"]
    assert torch.equal(seq_a.values, original_sequence)
    assert torch.equal(seq_a.lengths, original_lengths)
    assert torch.equal(augmented.inputs.user.dense_values, original_dense)


def test_augmentation_is_reproducible_with_fixed_generator_seed() -> None:
    batch = _make_batch()
    pipeline_config = PCVRDataPipelineConfig(
        transforms=(
            PCVRSequenceCropConfig(
                views_per_row=2,
                seq_window_mode="rolling",
                seq_window_min_len=1,
            ),
            PCVRFeatureMaskConfig(probability=0.3),
            PCVRDomainDropoutConfig(probability=0.2),
        ),
    )
    pipeline = PCVRDataPipeline(transforms=build_pcvr_batch_transforms(pipeline_config))

    first = pipeline.apply_transforms(
        batch, generator=torch.Generator().manual_seed(99)
    )
    second = pipeline.apply_transforms(
        batch, generator=torch.Generator().manual_seed(99)
    )

    _assert_batch_equal(first, second)


def test_lru_batch_cache_returns_isolated_clones() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="lru", max_batches=1)
    )
    cache.put(("file", 0, 0), _make_batch())

    cached = cache.get(("file", 0, 0))
    assert cached is not None
    cached.inputs.user.int_values[0, 0] = 999

    cached_again = cache.get(("file", 0, 0))
    assert cached_again is not None
    assert cached_again.inputs.user.int_values[0, 0].item() == 1


def test_fifo_batch_cache_uses_configured_eviction_policy() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="fifo", max_batches=2)
    )

    cache.put(("file", 0, 0), _make_batch())
    cache.put(("file", 0, 1), _make_batch())
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), _make_batch())

    assert cache.get(("file", 0, 0)) is None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is not None


def test_lfu_batch_cache_uses_configured_eviction_policy() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="lfu", max_batches=2)
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _make_batch())
    cache.put(("file", 0, 1), _make_batch())
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), _make_batch())

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is None
    assert cache.get(("file", 0, 2)) is not None
    stats = cache.stats()
    assert stats["native_cache_active"] is True
    assert stats["effective_policy"] == "lfu"


def test_rr_batch_cache_uses_native_index() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="rr", max_batches=2)
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _make_batch())
    cache.put(("file", 0, 1), _make_batch())
    cache.put(("file", 0, 2), _make_batch())

    assert len(cache) == 2
    stats = cache.stats()
    assert stats["native_cache_active"] is True
    assert stats["effective_policy"] == "rr"


def test_data_pipeline_keeps_explicit_empty_cache_instance() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )

    pipeline = PCVRDataPipeline(cache=cache)

    assert pipeline.cache is cache
    assert pipeline.cache._opt_enabled is True


def test_data_pipeline_materialize_composes_cache_preprocess_and_stages() -> None:
    events: list[str] = []

    class MarkStage:
        name = "mark"

        def __call__(self, batch, *, generator):
            del generator
            events.append("stage")
            return batch

    def factory():
        events.append("factory")
        return _make_batch()

    def preprocess(batch):
        events.append("preprocess")
        return batch

    pipeline = PCVRDataPipeline(cache=PCVRMemoryBatchCache(enabled=True, max_batches=1), stages=(MarkStage(),))

    first = pipeline.materialize(("file", 0, 0), factory, preprocess=preprocess)
    second = pipeline.materialize(("file", 0, 0), factory, preprocess=preprocess)

    assert first is not None and first.label.tolist() == [1, 0]
    assert second is not None and second.label.tolist() == [1, 0]
    assert events == ["factory", "preprocess", "stage", "preprocess", "stage"]


def test_concat_batch_joins_rows_and_user_ids() -> None:
    batch_a = _make_batch()
    batch_b = _make_batch()

    merged = concat_pcvr_batches([batch_a, batch_b])

    assert merged.user_id == ["u0", "u1", "u0", "u1"]
    assert merged.label.tolist() == [1, 0, 1, 0]


def test_opt_batch_cache_evicts_farthest_future_key() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key in (("file", 0, 0), ("file", 0, 1), ("file", 0, 2)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    stats = cache.stats()
    assert stats["opt_active"] is True
    assert stats["native_cache_active"] is True
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 4


def test_opt_batch_cache_supports_repeated_step_trace() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    trace = [
        ("file", 0, 0),
        ("file", 0, 1),
        ("file", 0, 0),
        ("file", 0, 2),
        ("file", 0, 0),
    ]
    cache.configure_access_trace(trace, cyclic=False)

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), _make_batch())
    assert cache.get(("file", 0, 1)) is None
    cache.put(("file", 0, 1), _make_batch())
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _make_batch())

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    stats = cache.stats()
    assert stats["opt_active"] is True
    assert stats["native_cache_active"] is True
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 5


def test_opt_batch_cache_skips_clone_for_rejected_candidate(monkeypatch) -> None:
    clone_calls = 0
    original_clone = cache_module.clone_pcvr_batch

    def counted_clone(batch):
        nonlocal clone_calls
        clone_calls += 1
        return original_clone(batch)

    monkeypatch.setattr(cache_module, "clone_pcvr_batch", counted_clone)
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key in (("file", 0, 0), ("file", 0, 1)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    assert clone_calls == 2
    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _make_batch())

    assert clone_calls == 2
    assert len(cache) == 2


def test_opt_batch_cache_rejects_candidate_without_future_use() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key in (("file", 0, 0), ("file", 0, 1)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _make_batch())

    assert len(cache) == 2
    assert cache.get(("file", 0, 2)) is None


def test_opt_batch_cache_reconfigures_same_trace() -> None:
    trace = [
        ("file", 0, 0),
        ("file", 0, 1),
        ("file", 0, 2),
    ]
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(trace)

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), _make_batch())

    cache.configure_access_trace(trace)

    assert cache.get(("file", 0, 0)) is not None
    assert cache._access_count == 1


def test_opt_batch_cache_window_restart_keeps_payloads() -> None:
    cache = PCVRMemoryBatchCache.from_config(
        PCVRDataCacheConfig(mode="opt", max_batches=2)
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 0),
        ],
        cyclic=False,
        key_universe=[("file", 0, 0), ("file", 0, 1), ("file", 0, 2)],
    )

    for key in (("file", 0, 0), ("file", 0, 1)):
        assert cache.get(key) is None
        cache.put(key, _make_batch())

    # A new trace window (e.g. next training sweep) over the same key universe
    # must keep the cached payloads instead of dropping and re-warming them.
    cache.configure_access_trace(
        [
            ("file", 0, 2),
            ("file", 0, 1),
            ("file", 0, 0),
        ],
        cyclic=False,
        key_universe=[("file", 0, 0), ("file", 0, 1), ("file", 0, 2)],
    )

    assert len(cache) == 2
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    assert cache.stats()["trace_length"] == 3
    assert cache._trace_positions_by_key == {2: (0,), 1: (1,), 0: (2,)}


def test_shared_opt_batch_cache_evicts_farthest_future_key() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
            ("file", 0, 3),
        ]
    )

    for key, value in zip(
        (("file", 0, 0), ("file", 0, 1), ("file", 0, 2)),
        (1, 2, 3),
        strict=True,
    ):
        assert cache.get(key) is None
        cache.put(key, _label_only_batch([value, value + 10]))

    cached_0 = cache.get(("file", 0, 0))
    cached_1 = cache.get(("file", 0, 1))
    cached_2 = cache.get(("file", 0, 2))

    assert cached_0 is not None
    assert cached_0.label.tolist() == [1, 11]
    assert cached_1 is not None
    assert cached_1.label.tolist() == [2, 12]
    assert cached_2 is None


def test_shared_opt_batch_cache_window_restart_keeps_slots() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    key_universe = [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    cache.configure_access_trace(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 0)],
        cyclic=False,
        key_universe=key_universe,
    )

    for key, value in zip(
        (("file", 0, 0), ("file", 0, 1)),
        (1, 2),
        strict=True,
    ):
        assert cache.get(key) is None
        cache.put(key, _label_only_batch([value, value + 10]))

    # A new trace window over the same key universe must keep shared slots.
    cache.configure_access_trace(
        [("file", 0, 2), ("file", 0, 1), ("file", 0, 0)],
        cyclic=False,
        key_universe=key_universe,
    )

    assert len(cache) == 2
    cached_0 = cache.get(("file", 0, 0))
    cached_1 = cache.get(("file", 0, 1))
    assert cached_0 is not None
    assert cached_0.label.tolist() == [1, 11]
    assert cached_1 is not None
    assert cached_1.label.tolist() == [2, 12]
    assert cache.get(("file", 0, 2)) is None
    assert cache.stats()["trace_length"] == 3


def test_shared_opt_batch_cache_supports_repeated_step_trace() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_access_trace(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 0),
            ("file", 0, 2),
            ("file", 0, 0),
        ],
        cyclic=False,
        key_universe=[("file", 0, 0), ("file", 0, 1), ("file", 0, 2)],
    )

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    assert cache.get(("file", 0, 1)) is None
    cache.put(("file", 0, 1), _label_only_batch([2, 12]))
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 2)) is None
    cache.put(("file", 0, 2), _label_only_batch([3, 13]))

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is None
    stats = cache.stats()
    assert stats["native_opt_active"] is True
    assert stats["trace_length"] == 5


def test_shared_opt_batch_cache_requires_trace_keys() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="opt",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_access_trace([("file", 0, 0)])

    assert cache.get(("file", 0, 0)) is None
    try:
        cache.get(("file", 0, 1))
    except KeyError as exc:
        assert "missing from configured access trace" in str(exc)
    else:
        raise AssertionError("expected KeyError for an untraced OPT cache key")


def test_shared_lru_batch_cache_reuses_slot_for_existing_key() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])

    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    cache.put(("file", 0, 0), _label_only_batch([2, 12]))

    cached = cache.get(("file", 0, 0))

    assert len(cache) == 1
    assert cached is not None
    assert cached.label.tolist() == [2, 12]


def test_shared_batch_cache_treats_busy_slot_as_miss() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])
    cache.put(("file", 0, 0), _label_only_batch([1, 11]))

    cache._slot_versions[0] += 1
    cached = cache.get(("file", 0, 0))

    assert cached is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_shared_batch_cache_preserves_per_row_user_ids_round_trip() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
        user_id_max_bytes=64,
    )
    cache.configure_key_universe([("file", 0, 0), ("file", 0, 1)])

    first = _label_only_batch_with_user_ids([1, 0], ["u0", "u1"])
    cache.put(("file", 0, 0), first)
    second = _label_only_batch([7, 8])
    cache.put(("file", 0, 1), second)

    cached_first = cache.get(("file", 0, 0))
    cached_second = cache.get(("file", 0, 1))

    assert cached_first is not None
    assert cached_first.user_id == ["u0", "u1"]
    assert cached_first.label.tolist() == [1, 0]
    assert cached_second is not None
    assert cached_second.user_id == []
    assert cached_second.label.tolist() == [7, 8]


def test_shared_batch_cache_rejects_user_ids_without_storage_budget() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
        user_id_max_bytes=0,
    )
    cache.configure_key_universe([("file", 0, 0)])

    with pytest.raises(ValueError, match="user_id_max_bytes"):
        cache.put(("file", 0, 0), _label_only_batch_with_user_ids([1, 0], ["u0", "u1"]))


def test_shared_batch_cache_rejects_user_ids_exceeding_budget() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
        user_id_max_bytes=2,
    )
    cache.configure_key_universe([("file", 0, 0)])

    with pytest.raises(ValueError, match="exceeds user_id_max_bytes"):
        cache.put(("file", 0, 0), _label_only_batch_with_user_ids([1, 0], ["u0", "u1"]))


def test_shared_batch_cache_discards_payload_when_version_changes(monkeypatch) -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])
    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    original_materialize = cache._materialize_slot

    def materialize_and_invalidate(slot_index: int, row_count: int):
        batch = original_materialize(slot_index, row_count)
        cache._slot_versions[slot_index] += 2
        return batch

    monkeypatch.setattr(cache, "_materialize_slot", materialize_and_invalidate)

    cached = cache.get(("file", 0, 0))

    assert cached is None
    stats = cache.stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 1


def test_shared_fifo_batch_cache_uses_native_index() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="fifo",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    cache.put(("file", 0, 1), _label_only_batch([2, 12]))
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), _label_only_batch([3, 13]))

    assert cache.get(("file", 0, 0)) is None
    assert cache.get(("file", 0, 1)) is not None
    assert cache.get(("file", 0, 2)) is not None
    assert cache.stats()["native_cache_active"] is True


def test_shared_lfu_batch_cache_uses_native_index() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="lfu",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    cache.put(("file", 0, 1), _label_only_batch([2, 12]))
    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 0)) is not None
    cache.put(("file", 0, 2), _label_only_batch([3, 13]))

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is None
    assert cache.get(("file", 0, 2)) is not None
    assert cache.stats()["native_cache_active"] is True


def test_shared_rr_batch_cache_uses_native_index() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="rr",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [("file", 0, 0), ("file", 0, 1), ("file", 0, 2)]
    )

    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    cache.put(("file", 0, 1), _label_only_batch([2, 12]))
    cache.put(("file", 0, 2), _label_only_batch([3, 13]))

    assert len(cache) == 2
    assert cache.stats()["native_cache_active"] is True


def test_shared_lru_batch_cache_uses_key_universe_and_tracks_hits() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=2,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe(
        [
            ("file", 0, 0),
            ("file", 0, 1),
            ("file", 0, 2),
        ]
    )

    assert cache.get(("file", 0, 0)) is None
    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    cache.put(("file", 0, 1), _label_only_batch([2, 12]))

    cached = cache.get(("file", 0, 0))
    assert cached is not None
    assert cached.label.tolist() == [1, 11]

    cache.put(("file", 0, 2), _label_only_batch([3, 13]))

    assert cache.get(("file", 0, 0)) is not None
    assert cache.get(("file", 0, 1)) is None
    assert cache.get(("file", 0, 2)) is not None

    stats = cache.stats()
    assert stats["shared_lru_active"] is True
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.6


def test_shared_batch_cache_overwrites_partial_slot_without_stale_rows() -> None:
    cache = PCVRSharedBatchCache(
        enabled=True,
        max_batches=1,
        policy="lru",
        tensor_specs=_shared_label_specs(2),
        static_values={"_seq_domains": []},
    )
    cache.configure_key_universe([("file", 0, 0)])

    cache.put(("file", 0, 0), _label_only_batch([1, 11]))
    cache.put(("file", 0, 0), _label_only_batch([2]))

    cached = cache.get(("file", 0, 0))

    assert cached is not None
    assert cached.label.tolist() == [2]


def test_strict_time_filter_removes_future_sequence_events(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    schema = {
        "format": "raw_parquet",
        "user_int": [[1, 10, 1]],
        "item_int": [[2, 10, 1]],
        "user_dense": [[3, 2]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 1000], [11, 100]],
            }
        },
    }
    schema_path.write_text(dumps(schema), encoding="utf-8")
    table = pa.table(
        {
            "timestamp": [100],
            "label_type": [2],
            "user_id": ["u0"],
            "user_int_feats_1": [1],
            "item_int_feats_2": [2],
            "user_dense_feats_3": [[0.1, 0.2]],
            "domain_a_seq_10": [[10, 150, 90]],
            "domain_a_seq_11": [[1, 2, 3]],
        }
    )
    pq.write_table(table, parquet_path, row_group_size=1)
    dataset = PCVRParquetDataset(
        parquet_path=str(parquet_path),
        schema_path=str(schema_path),
        batch_size=1,
        seq_max_lens={"seq_a": 3},
        shuffle=False,
        buffer_batches=0,
        data_pipeline_config=PCVRDataPipelineConfig(
            transforms=(PCVRSequenceCropConfig(),),
            strict_time_filter=True,
        ),
    )

    batch = next(iter(dataset))

    seq_a = batch.inputs.sequences["seq_a"]
    assert seq_a.lengths.tolist() == [2]
    assert seq_a.values.tolist() == [[[1, 3, 0]]]
    assert seq_a.timestamps[0, :2].gt(0).all()
    assert seq_a.timestamps[0, 2].item() == 0


def test_dataset_logs_schema_payload_with_dataset_role(tmp_path: Path, log_capture) -> None:
    schema_path = tmp_path / "schema.json"
    parquet_path = tmp_path / "demo.parquet"
    schema = {
        "format": "raw_parquet",
        "user_int": [[1, 10, 1]],
        "item_int": [[2, 10, 1]],
        "user_dense": [[3, 2]],
        "seq": {
            "seq_a": {
                "prefix": "domain_a_seq",
                "ts_fid": 10,
                "features": [[10, 1000], [11, 100]],
            }
        },
    }
    schema_path.write_text(dumps(schema), encoding="utf-8")
    table = pa.table(
        {
            "timestamp": [100],
            "label_type": [2],
            "user_id": ["u0"],
            "user_int_feats_1": [1],
            "item_int_feats_2": [2],
            "user_dense_feats_3": [[0.1, 0.2]],
            "domain_a_seq_10": [[10]],
            "domain_a_seq_11": [[1]],
        }
    )
    pq.write_table(table, parquet_path, row_group_size=1)

    with log_capture.at_level(logging.INFO):
        PCVRParquetDataset(
            parquet_path=str(parquet_path),
            schema_path=str(schema_path),
            batch_size=1,
            seq_max_lens={"seq_a": 1},
            shuffle=False,
            buffer_batches=0,
            dataset_role="train",
        )

    assert "Loaded PCVR schema for train dataset" in log_capture.text
    assert str(schema_path.resolve()) in log_capture.text
    assert "PCVR train schema payload" in log_capture.text
    payload_message = next(
        record.getMessage()
        for record in log_capture.records
        if record.getMessage().startswith("PCVR train schema payload: ")
    )
    assert "\n" not in payload_message
    assert '"user_int":[{"fid":1,"vocab_size":10,"dim":1}]' in payload_message
    assert '"prefix":"domain_a_seq"' in payload_message
