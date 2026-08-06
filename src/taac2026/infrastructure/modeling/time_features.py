"""Model-side time features derived from raw event timestamps.

Time buckets and sequence statistics are pure functions of the canonical batch
(``PCVRSequenceInput`` + request timestamp); the data pipeline never computes
them.
"""

from __future__ import annotations

import numpy as np
import torch

SEQUENCE_STATS_DIM = 6

BUCKET_BOUNDARIES = np.array(
    [
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
        120, 180, 240, 300, 360, 420, 480, 540, 600, 900, 1200, 1500, 1800,
        2100, 2400, 2700, 3000, 3300, 3600, 5400, 7200, 9000, 10800, 12600,
        14400, 16200, 18000, 19800, 21600, 32400, 43200, 54000, 64800, 75600,
        86400, 172800, 259200, 345600, 432000, 518400, 604800, 1123200,
        1641600, 2160000, 2592000, 4320000, 6048000, 7776000, 11664000,
        15552000, 31536000,
    ],
    dtype=np.int64,
)

NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


def compute_sequence_time_buckets(
    seq_timestamps: torch.Tensor,
    request_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Bucket gap times between request and each event, matching converter semantics.

    Padded (zero) event timestamps map to bucket 0.
    """
    if seq_timestamps.shape[1] == 0:
        return seq_timestamps.new_zeros(seq_timestamps.shape, dtype=torch.long)
    bounds = torch.as_tensor(BUCKET_BOUNDARIES, dtype=torch.long, device=seq_timestamps.device)
    time_diff = torch.clamp(request_timestamps[:, None] - seq_timestamps, min=0)
    raw_buckets = torch.searchsorted(bounds, time_diff)
    buckets = torch.clamp(raw_buckets, max=len(BUCKET_BOUNDARIES) - 1) + 1
    buckets = torch.where(seq_timestamps == 0, torch.zeros_like(buckets), buckets)
    return buckets


def compute_sequence_stats(
    sequence: torch.Tensor,
    lengths: torch.Tensor,
    seq_timestamps: torch.Tensor,
    request_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Compute the 6-dim sequence statistics from the canonical batch.

    Columns: [length, active_events, unique_events, dup_ratio, nonzero_fraction,
    last_gap_bucket]. Rows without active events stay zero.
    """
    batch_size, feature_count, max_len = sequence.shape
    stats = sequence.new_zeros((batch_size, SEQUENCE_STATS_DIM), dtype=torch.float32)
    if feature_count <= 0:
        return stats
    length_values = torch.clamp(lengths, min=0, max=max_len)
    positions = torch.arange(max_len, device=sequence.device)
    active = (sequence > 0).any(dim=1) & (positions[None, :] < length_values[:, None])
    active_count = active.sum(dim=1)
    has_events = active_count > 0

    events = sequence.transpose(1, 2)  # [B, L, F]
    masked = torch.where(active.unsqueeze(-1), events, torch.full_like(events, -1))
    order = torch.arange(max_len, device=sequence.device).expand(batch_size, -1).clone()
    for feature_index in range(feature_count - 1, -1, -1):
        keys = masked.gather(1, order.unsqueeze(-1).expand(-1, -1, feature_count))[:, :, feature_index]
        permutation = torch.argsort(keys, dim=1, stable=True)
        order = order.gather(1, permutation)
    sorted_masked = masked.gather(1, order.unsqueeze(-1).expand(-1, -1, feature_count))
    sorted_active = active.gather(1, order)
    differs_from_previous = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=torch.bool, device=sequence.device),
            (sorted_masked[:, 1:] != sorted_masked[:, :-1]).any(dim=-1),
        ],
        dim=1,
    )
    unique_count = (differs_from_previous & sorted_active).sum(dim=1)

    nonzero_fraction = (sequence > 0).sum(dim=(1, 2)) / torch.maximum(
        length_values * feature_count, torch.ones_like(length_values, dtype=torch.long)
    )
    buckets = compute_sequence_time_buckets(seq_timestamps, request_timestamps)
    last_index = (length_values - 1).clamp(min=0)
    last_bucket = buckets[torch.arange(batch_size, device=sequence.device), last_index]

    stats[:, 0] = torch.where(has_events, length_values.to(torch.float32), stats[:, 0])
    stats[:, 1] = torch.where(has_events, active_count.to(torch.float32), stats[:, 1])
    stats[:, 2] = torch.where(has_events, unique_count.to(torch.float32), stats[:, 2])
    stats[:, 3] = torch.where(
        has_events,
        1.0 - unique_count.to(torch.float32) / torch.maximum(active_count.to(torch.float32), torch.ones_like(stats[:, 0])),
        stats[:, 3],
    )
    stats[:, 4] = torch.where(has_events, nonzero_fraction.to(torch.float32), stats[:, 4])
    stats[:, 5] = torch.where(has_events & (length_values > 0), last_bucket.to(torch.float32), stats[:, 5])
    return stats


__all__ = [
    "BUCKET_BOUNDARIES",
    "NUM_TIME_BUCKETS",
    "SEQUENCE_STATS_DIM",
    "compute_sequence_stats",
    "compute_sequence_time_buckets",
]
