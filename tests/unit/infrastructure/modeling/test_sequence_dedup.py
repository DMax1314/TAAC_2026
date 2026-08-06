from __future__ import annotations

import torch

from taac2026.infrastructure.modeling.sequence import deduplicate_sequence_events


def _dedup(values: list[list[list[int]]], lengths: list[int], timestamps: list[list[int]]) -> tuple[list[list[list[int]]], list[int], list[list[int]]]:
    result = deduplicate_sequence_events(
        torch.tensor(values, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(timestamps, dtype=torch.long),
    )
    return (
        result[0].tolist(),
        result[1].tolist(),
        result[2].tolist(),
    )


def test_dedup_keeps_last_occurrence_and_compacts_in_original_order() -> None:
    # Duplicate signature (2, 2) at positions 1 and 2: the last occurrence wins.
    values, lengths, timestamps = _dedup(
        [
            [[1, 2, 2, 3], [1, 2, 2, 3]],
            [[4, 4, 0, 0], [5, 5, 0, 0]],
        ],
        [4, 2],
        [
            [10, 20, 30, 40],
            [95, 96, 0, 0],
        ],
    )
    assert values == [
        [[1, 2, 3, 0], [1, 2, 3, 0]],
        [[4, 0, 0, 0], [5, 0, 0, 0]],
    ]
    assert lengths == [3, 1]
    assert timestamps == [
        [10, 30, 40, 0],
        [96, 0, 0, 0],
    ]


def test_dedup_keeps_last_occurrence_matching_numpy_semantics() -> None:
    # [7, 7, 9] -> [7, 9]: length 2, timestamps follow the kept (last) events.
    values, lengths, timestamps = _dedup(
        [[[7, 7, 9]]],
        [3],
        [[100, 200, 300]],
    )
    assert values == [[[7, 9, 0]]]
    assert lengths == [2]
    assert timestamps == [[200, 300, 0]]


def test_dedup_leaves_short_rows_untouched() -> None:
    values, lengths, timestamps = _dedup(
        [[[5, 0, 0], [6, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        [1, 0],
        [[50, 0, 0], [0, 0, 0]],
    )
    assert values == [[[5, 0, 0], [6, 0, 0]], [[0, 0, 0], [0, 0, 0]]]
    assert lengths == [1, 0]
    assert timestamps == [[50, 0, 0], [0, 0, 0]]


def test_dedup_empties_rows_with_only_inactive_events() -> None:
    values, lengths, timestamps = _dedup(
        [[[0, 0, 0], [0, 0, 0]], [[1, 2, 2], [3, 4, 4]]],
        [3, 3],
        [[0, 0, 0], [10, 20, 30]],
    )
    assert values == [[[0, 0, 0], [0, 0, 0]], [[1, 2, 0], [3, 4, 0]]]
    assert lengths == [0, 2]
    assert timestamps == [[0, 0, 0], [10, 30, 0]]


def test_dedup_without_features_empties_long_rows() -> None:
    values = torch.zeros(2, 0, 4, dtype=torch.long)
    lengths = torch.tensor([4, 1], dtype=torch.long)
    timestamps = torch.tensor([[1, 2, 3, 4], [9, 0, 0, 0]], dtype=torch.long)
    deduped_values, deduped_lengths, deduped_timestamps = deduplicate_sequence_events(
        values, lengths, timestamps
    )
    assert deduped_values.shape == (2, 0, 4)
    assert deduped_lengths.tolist() == [0, 1]
    assert deduped_timestamps.tolist() == [[0, 0, 0, 0], [9, 0, 0, 0]]


def test_dedup_unique_events_are_unchanged() -> None:
    values, lengths, timestamps = _dedup(
        [[[1, 2, 3], [4, 5, 6]]],
        [3],
        [[10, 20, 30]],
    )
    assert values == [[[1, 2, 3], [4, 5, 6]]]
    assert lengths == [3]
    assert timestamps == [[10, 20, 30]]


def test_dedup_is_differentiable_agnostic_and_empty_batch_safe() -> None:
    batch = torch.tensor([[[1, 1, 2]]], dtype=torch.long)
    lengths = torch.tensor([3], dtype=torch.long)
    timestamps = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        deduped_values, deduped_lengths, deduped_timestamps = deduplicate_sequence_events(
            batch, lengths, timestamps
        )
    assert deduped_values.tolist() == [[[1, 2, 0]]]
    assert deduped_lengths.tolist() == [2]
    assert deduped_timestamps.tolist() == [[2, 3, 0]]

    empty = deduplicate_sequence_events(
        torch.zeros(0, 1, 4, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, 4, dtype=torch.long),
    )
    assert all(tensor.shape[0] == 0 for tensor in empty)

    # Zero-feature sequences keep short rows and empty longer rows.
    no_features = deduplicate_sequence_events(
        torch.zeros(1, 0, 4, dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    )
    assert no_features[1].tolist() == [0]
    assert no_features[2].tolist() == [[0, 0, 0, 0]]
