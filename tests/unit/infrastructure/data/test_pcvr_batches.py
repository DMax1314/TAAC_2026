"""Structured PCVR batch carrier and tensor utility tests."""

from __future__ import annotations

import torch

from taac2026.infrastructure.data.batches import (
    PCVRBatch,
    PCVREntityInput,
    PCVRModelInput,
    PCVRSequenceInput,
    clone_pcvr_batch,
    concat_pcvr_batches,
    get_pcvr_batch_tensor,
    pcvr_batch_from_parts,
    pcvr_batch_row_count,
    pcvr_tensor_paths,
    repeat_pcvr_rows,
    take_pcvr_rows,
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
        values=torch.tensor([[[1, 2, 0], [2, 3, 0]], [[7, 8, 9], [1, 2, 3]]], dtype=torch.long),
        lengths=torch.tensor([2, 3], dtype=torch.long),
        timestamps=torch.tensor([[10, 20, 0], [70, 80, 90]], dtype=torch.long),
    )
    inputs = PCVRModelInput(
        user=user,
        item=item,
        sequences={"seq_a": seq_a},
        request_timestamp=torch.tensor([100, 200], dtype=torch.long),
    )
    return PCVRBatch(inputs=inputs, label=torch.tensor([1, 0], dtype=torch.long), user_id=["u0", "u1"])


def test_pcvr_tensor_paths_cover_all_structured_tensors() -> None:
    paths = pcvr_tensor_paths(["seq_a", "seq_b"])

    assert paths["user_int_values"] == ("inputs", "user", "int_values")
    assert paths["user_int_missing_mask"] == ("inputs", "user", "int_missing_mask")
    assert paths["user_dense_values"] == ("inputs", "user", "dense_values")
    assert paths["user_dense_missing_mask"] == ("inputs", "user", "dense_missing_mask")
    assert paths["item_int_values"] == ("inputs", "item", "int_values")
    assert paths["item_int_missing_mask"] == ("inputs", "item", "int_missing_mask")
    assert paths["item_dense_values"] == ("inputs", "item", "dense_values")
    assert paths["item_dense_missing_mask"] == ("inputs", "item", "dense_missing_mask")
    assert paths["request_timestamp"] == ("inputs", "request_timestamp")
    assert paths["label"] == ("label",)
    assert paths["seq_a_values"] == ("inputs", "sequences", "seq_a", "values")
    assert paths["seq_a_lengths"] == ("inputs", "sequences", "seq_a", "lengths")
    assert paths["seq_a_timestamps"] == ("inputs", "sequences", "seq_a", "timestamps")
    assert paths["seq_b_values"] == ("inputs", "sequences", "seq_b", "values")


def test_get_pcvr_batch_tensor_resolves_entity_and_sequence_tensors() -> None:
    batch = _make_batch()

    assert torch.equal(
        get_pcvr_batch_tensor(batch, ("inputs", "user", "int_values")),
        batch.inputs.user.int_values,
    )
    assert torch.equal(
        get_pcvr_batch_tensor(batch, ("inputs", "sequences", "seq_a", "timestamps")),
        batch.inputs.sequences["seq_a"].timestamps,
    )
    assert torch.equal(get_pcvr_batch_tensor(batch, ("label",)), batch.label)


def test_pcvr_batch_from_parts_round_trips_through_flat_parts() -> None:
    batch = _make_batch()
    paths = pcvr_tensor_paths(["seq_a"])
    tensors = {key: get_pcvr_batch_tensor(batch, path) for key, path in paths.items()}

    rebuilt = pcvr_batch_from_parts(tensors, seq_domains=["seq_a"], user_id=["u0", "u1"])

    assert torch.equal(rebuilt.inputs.user.int_values, batch.inputs.user.int_values)
    assert torch.equal(rebuilt.inputs.sequences["seq_a"].values, batch.inputs.sequences["seq_a"].values)
    assert torch.equal(rebuilt.inputs.request_timestamp, batch.inputs.request_timestamp)
    assert torch.equal(rebuilt.label, batch.label)
    assert rebuilt.user_id == ["u0", "u1"]


def test_pcvr_batch_to_moves_all_tensors_to_device() -> None:
    batch = _make_batch()

    moved = batch.to(torch.device("cpu"))

    assert torch.equal(moved.inputs.user.int_values, batch.inputs.user.int_values)
    assert torch.equal(moved.inputs.sequences["seq_a"].timestamps, batch.inputs.sequences["seq_a"].timestamps)
    assert moved.user_id == batch.user_id


def test_clone_pcvr_batch_returns_independent_copy() -> None:
    batch = _make_batch()
    cloned = clone_pcvr_batch(batch)

    cloned.inputs.user.int_values[0, 0] = 999
    cloned.inputs.sequences["seq_a"].values[0, 0, 0] = 999

    assert batch.inputs.user.int_values[0, 0].item() == 1
    assert batch.inputs.sequences["seq_a"].values[0, 0, 0].item() == 1
    assert cloned.user_id == ["u0", "u1"]


def test_repeat_pcvr_rows_interleaves_rows() -> None:
    batch = _make_batch()

    repeated = repeat_pcvr_rows(batch, repeats=2)

    assert repeated.label.tolist() == [1, 1, 0, 0]
    assert repeated.user_id == ["u0", "u0", "u1", "u1"]
    assert repeated.inputs.user.int_values.tolist() == [[1, 2], [1, 2], [3, 4], [3, 4]]
    assert repeated.inputs.sequences["seq_a"].lengths.tolist() == [2, 2, 3, 3]
    assert repeated.inputs.request_timestamp.tolist() == [100, 100, 200, 200]


def test_take_pcvr_rows_selects_rows() -> None:
    batch = _make_batch()

    taken = take_pcvr_rows(batch, torch.tensor([1], dtype=torch.long))

    assert taken.label.tolist() == [0]
    assert taken.user_id == ["u1"]
    assert taken.inputs.user.int_values.tolist() == [[3, 4]]
    assert taken.inputs.sequences["seq_a"].values.tolist() == [[[7, 8, 9], [1, 2, 3]]]
    assert taken.inputs.sequences["seq_a"].lengths.tolist() == [3]


def test_concat_pcvr_batches_stacks_rows() -> None:
    batch = _make_batch()
    batch_b = _make_batch()

    merged = concat_pcvr_batches([batch, batch_b])

    assert merged.label.tolist() == [1, 0, 1, 0]
    assert merged.user_id == ["u0", "u1", "u0", "u1"]
    assert merged.inputs.sequences["seq_a"].values.shape == (4, 2, 3)
    assert merged.inputs.sequences["seq_a"].lengths.tolist() == [2, 3, 2, 3]
    assert merged.inputs.sequences["seq_a"].timestamps.tolist()[0] == [10, 20, 0]


def test_pcvr_batch_row_count_uses_label_rows() -> None:
    batch = _make_batch()

    assert pcvr_batch_row_count(batch) == 2
