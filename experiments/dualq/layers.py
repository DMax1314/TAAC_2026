"""DualQ-private data views and time feature derivations.

The model never reads parquet or dataset dictionaries: everything here is a
pure function of ``PCVRSchema`` (compiled at construction) and
``PCVRModelInput`` tensors (sliced at forward time).

* ``compile_pair_split`` partitions user int/dense columns into pair and
  non-pair views by fid, mirroring the source repository's dataset-side pair
  diversion.
* The ``compute_*`` functions re-derive the source dataset's per-position
  time features (``TS_FLOAT_DIM = 8``), per-domain time statistics
  (``TS_STAT_DIM = 6``), gap buckets and global-time features from raw event
  timestamps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from taac2026.api import PCVRSchema
from taac2026.infrastructure.modeling.time_features import BUCKET_BOUNDARIES

from .hyformer import TS_FLOAT_DIM, TS_STAT_DIM

GLOBAL_TIMEZONE_OFFSET_SECONDS = 8 * 3600


def parse_pair_feature_fids(value: str) -> list[int]:
    """Parse the comma-separated pair fid string into a list of ints."""
    return [int(part) for part in str(value).split(",") if part.strip()]


@dataclass(frozen=True, slots=True)
class PairSplitPlan:
    """Compiled raw-layout offsets for pair vs. non-pair user columns.

    Raw offsets index into the full ``PCVRModelInput`` tensors; the
    ``*_reduced`` properties index into the sliced non-pair tensors built by
    ``split_*``.
    """

    user_int_entries: tuple[tuple[int, int, int], ...]  # (fid, offset, length)
    pair_int_entries: tuple[tuple[int, int, int], ...]
    user_dense_entries: tuple[tuple[int, int, int], ...]  # (fid, offset, dim)
    pair_dense_entries: tuple[tuple[int, int, int], ...]

    @property
    def user_int_reduced(self) -> tuple[tuple[int, int, int], ...]:
        return _reduced_offsets(self.user_int_entries)

    @property
    def user_dense_reduced(self) -> tuple[tuple[int, int, int], ...]:
        return _reduced_offsets(self.user_dense_entries)

    @property
    def pair_int_reduced(self) -> tuple[tuple[int, int, int], ...]:
        return _reduced_offsets(self.pair_int_entries)

    @property
    def pair_dense_reduced(self) -> tuple[tuple[int, int, int], ...]:
        return _reduced_offsets(self.pair_dense_entries)

    @property
    def pair_feature_fids(self) -> list[int]:
        return [fid for fid, _, _ in self.pair_int_entries]

    def split_user_int(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice full user int values into (non-pair, pair) tensors."""
        user_parts = [
            values[:, offset : offset + length]
            for _, offset, length in self.user_int_entries
        ]
        pair_parts = [
            values[:, offset : offset + length]
            for _, offset, length in self.pair_int_entries
        ]
        return _cat(user_parts, values), _cat(pair_parts, values)

    def split_user_dense(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice full user dense values into (non-pair, log1p pair) tensors.

        The source dataset log1p-compresses the dense counts of pair fids
        below 89 (the head feature block) and keeps fids 89-91 raw.
        """
        user_parts = [
            values[:, offset : offset + dim]
            for _, offset, dim in self.user_dense_entries
        ]
        pair_parts = []
        for fid, offset, dim in self.pair_dense_entries:
            part = values[:, offset : offset + dim]
            if fid < 89:
                part = torch.log1p(part.clamp_min(0.0))
            pair_parts.append(part)
        return _cat(user_parts, values), _cat(pair_parts, values)


def _reduced_offsets(
    entries: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Rebase (fid, offset, width) entries onto a contiguous layout."""
    reduced: list[tuple[int, int, int]] = []
    cursor = 0
    for fid, _offset, width in entries:
        reduced.append((fid, cursor, width))
        cursor += width
    return tuple(reduced)


def _cat(parts: list[torch.Tensor], values: torch.Tensor) -> torch.Tensor:
    if parts:
        return torch.cat(parts, dim=1)
    return values.new_zeros(values.shape[0], 0)


def compile_pair_split(schema: PCVRSchema, pair_fids: list[int]) -> PairSplitPlan:
    """Compile pair/non-pair slices from a raw parquet schema."""
    pair_set = set(pair_fids)

    user_int_entries: list[tuple[int, int, int]] = []
    pair_int_entries: list[tuple[int, int, int]] = []
    raw_offset = 0
    for column in schema.user_int:
        entry = (column.fid, raw_offset, column.dim)
        if column.fid in pair_set:
            pair_int_entries.append(entry)
        else:
            user_int_entries.append(entry)
        raw_offset += column.dim

    user_dense_entries: list[tuple[int, int, int]] = []
    pair_dense_entries: list[tuple[int, int, int]] = []
    raw_offset = 0
    for column in schema.user_dense:
        entry = (column.fid, raw_offset, column.dim)
        if column.fid in pair_set:
            pair_dense_entries.append(entry)
        else:
            user_dense_entries.append(entry)
        raw_offset += column.dim

    return PairSplitPlan(
        user_int_entries=tuple(user_int_entries),
        pair_int_entries=tuple(pair_int_entries),
        user_dense_entries=tuple(user_dense_entries),
        pair_dense_entries=tuple(pair_dense_entries),
    )


def _sequence_positions(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def compute_sequence_gap_buckets(
    timestamps: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Bucket consecutive-event time gaps, mirroring the source converter.

    Position ``i`` carries the absolute gap between events ``i-1`` and ``i``
    (positions 0 and padded positions stay 0).
    """
    bounds = torch.as_tensor(BUCKET_BOUNDARIES, dtype=torch.long, device=timestamps.device)
    shifted = torch.zeros_like(timestamps)
    shifted[:, 1:] = timestamps[:, :-1]
    gaps = (timestamps - shifted).abs()
    raw = torch.searchsorted(bounds, gaps)
    buckets = torch.clamp(raw, max=len(BUCKET_BOUNDARIES) - 1) + 1
    valid = (timestamps > 0) & (shifted > 0) & _sequence_positions(lengths, timestamps.shape[1])
    return torch.where(valid, buckets, torch.zeros_like(buckets))


def _hour_decimal_cos_sin(timestamps: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hour_dec = ((timestamps % 86400).float() / 3600.0 * 10.0).round() / 10.0
    hour_dec = hour_dec.clamp(0.0, 23.9)
    angle = (2.0 * math.pi / 24.0) * hour_dec
    cos_val = torch.cos(angle)
    sin_val = torch.sin(angle)
    return torch.where(valid, cos_val, torch.zeros_like(cos_val)), torch.where(
        valid, sin_val, torch.zeros_like(sin_val)
    )


def compute_sequence_ts_float(
    timestamps: torch.Tensor,
    lengths: torch.Tensor,
    request_timestamps: torch.Tensor,
    domain: str,
) -> torch.Tensor:
    """Derive the 8-dim per-position time features of the source dataset.

    Columns: [log1p(diff_days), domain-scaled diff, log1p(diff_hours),
    event hour cos, event hour sin, event dow cos, event dow sin,
    log1p/clamped next-event gap].
    """
    batch, max_len = timestamps.shape
    out = timestamps.new_zeros((batch, TS_FLOAT_DIM, max_len), dtype=torch.float32)
    valid = (timestamps > 0) & _sequence_positions(lengths, max_len)
    diff = (request_timestamps[:, None] - timestamps).clamp(min=0).float()
    d = torch.where(valid, diff, torch.zeros_like(diff))
    d_days = d / 86400.0

    out[:, 0, :] = torch.log1p(d_days)
    if domain == "seq_c":
        out[:, 1, :] = d_days / 30.0
    elif domain == "seq_d":
        out[:, 1, :] = d / 3600.0
    else:
        out[:, 1, :] = d_days
    out[:, 2, :] = torch.log1p(d / 3600.0)

    if domain != "seq_c":
        hour_cos, hour_sin = _hour_decimal_cos_sin(timestamps, valid)
        out[:, 3, :] = hour_cos
        out[:, 4, :] = hour_sin
        dow = (timestamps // 86400) % 7
        dow_cos = torch.cos(2.0 * math.pi * dow / 7.0)
        dow_sin = torch.sin(2.0 * math.pi * dow / 7.0)
        out[:, 5, :] = torch.where(valid, dow_cos, torch.zeros_like(dow_cos))
        out[:, 6, :] = torch.where(valid, dow_sin, torch.zeros_like(dow_sin))

    shifted = torch.zeros_like(timestamps)
    shifted[:, :-1] = timestamps[:, 1:]
    inter_s = torch.where(
        valid & (shifted > 0),
        (timestamps - shifted).clamp(min=0).float(),
        torch.zeros_like(timestamps, dtype=torch.float32),
    )
    if domain == "seq_c":
        out[:, 7, :] = torch.where(valid, inter_s / (86400.0 * 30), torch.zeros_like(inter_s))
    elif domain == "seq_d":
        out[:, 7, :] = torch.where(valid, torch.log1p(inter_s / 3600.0), torch.zeros_like(inter_s))
    else:
        out[:, 7, :] = torch.where(valid, torch.log1p(inter_s / 86400.0), torch.zeros_like(inter_s))
    return out


def compute_sequence_ts_stat(
    timestamps: torch.Tensor,
    lengths: torch.Tensor,
    request_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Derive the 6-dim per-domain time statistics of the source dataset.

    Columns: [log1p(max diff), log1p(min diff), log1p(mean diff), events
    within 15min, events within 1h, events within 1day].
    """
    batch, max_len = timestamps.shape
    stats = timestamps.new_zeros((batch, TS_STAT_DIM), dtype=torch.float32)
    valid = (timestamps > 0) & _sequence_positions(lengths, max_len)
    diff_f = torch.where(
        valid,
        (request_timestamps[:, None] - timestamps).clamp(min=0).float(),
        torch.zeros_like(timestamps, dtype=torch.float32),
    )
    n_valid = valid.sum(dim=1)
    has_valid = n_valid > 0

    max_v = torch.where(valid, diff_f, torch.full_like(diff_f, -1e9)).max(dim=1).values
    max_v = torch.where(has_valid, max_v, torch.zeros_like(max_v))
    min_v = torch.where(valid, diff_f, torch.full_like(diff_f, 1e9)).min(dim=1).values
    min_v = torch.where(has_valid, min_v, torch.zeros_like(min_v))
    mean_v = diff_f.sum(dim=1) / n_valid.clamp(min=1)
    mean_v = torch.where(has_valid, mean_v, torch.zeros_like(mean_v))

    stats[:, 0] = torch.log1p(max_v.clamp_min(0.0))
    stats[:, 1] = torch.log1p(min_v.clamp_min(0.0))
    stats[:, 2] = torch.log1p(mean_v.clamp_min(0.0))
    stats[:, 3] = (valid & (diff_f <= 900)).sum(dim=1).float()
    stats[:, 4] = (valid & (diff_f <= 3600)).sum(dim=1).float()
    stats[:, 5] = (valid & (diff_f <= 86400)).sum(dim=1).float()
    return stats


def build_global_time_features(request_timestamps: torch.Tensor) -> torch.Tensor:
    """Derive [hour, day-of-week, weekend] ids from request timestamps.

    Mirrors the source dataset: UTC+8 local time, hour in [1, 24],
    day-of-week in [1, 7], weekend flag in [1, 2].
    """
    local_ts = (request_timestamps + GLOBAL_TIMEZONE_OFFSET_SECONDS).clamp(min=0)
    days = local_ts // 86400
    hour = ((local_ts // 3600) % 24) + 1
    day_of_week = ((days + 3) % 7) + 1
    is_weekend = (day_of_week >= 6) + 1
    return torch.stack([hour, day_of_week, is_weekend], dim=1)


__all__ = [
    "GLOBAL_TIMEZONE_OFFSET_SECONDS",
    "PairSplitPlan",
    "build_global_time_features",
    "compile_pair_split",
    "compute_sequence_gap_buckets",
    "compute_sequence_ts_float",
    "compute_sequence_ts_stat",
    "parse_pair_feature_fids",
]
