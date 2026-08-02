from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch
from numpy.typing import NDArray

from taac2026.domain.schema import BUCKET_BOUNDARIES
from taac2026.infrastructure.data.schema_layout import PCVRSchemaLayout
from taac2026.infrastructure.logging import logger


SEQUENCE_STATS_DIM = 6


@dataclass(frozen=True, slots=True)
class IntColumnPlan:
    column_index: int
    dim: int
    output_offset: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class DenseColumnPlan:
    column_index: int
    dim: int
    output_offset: int


@dataclass(frozen=True, slots=True)
class SequenceSideColumnPlan:
    column_index: int
    slot: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class SequenceColumnPlan:
    domain: str
    max_len: int
    side_columns: tuple[SequenceSideColumnPlan, ...]
    timestamp_column_index: int | None


@dataclass(frozen=True, slots=True)
class PCVRColumnPlan:
    required_column_names: tuple[str, ...]
    column_indices: dict[str, int]
    user_int: tuple[IntColumnPlan, ...]
    item_int: tuple[IntColumnPlan, ...]
    user_dense: tuple[DenseColumnPlan, ...]
    sequences: dict[str, SequenceColumnPlan]

    def record_batch_columns(self) -> list[str] | None:
        return list(self.required_column_names) or None


def pad_list_offsets_values(
    offsets: NDArray[Any],
    values: NDArray[Any],
    *,
    row_count: int,
    width: int,
    dtype: np.dtype[Any] | type[np.generic],
) -> tuple[NDArray[Any], NDArray[np.int64]]:
    padded = np.zeros((row_count, width), dtype=dtype)
    if row_count <= 0 or width <= 0:
        return padded, np.zeros(row_count, dtype=np.int64)

    starts = np.asarray(offsets[:row_count], dtype=np.int64)
    ends = np.asarray(offsets[1 : row_count + 1], dtype=np.int64)
    raw_lengths = np.maximum(ends - starts, 0)
    lengths = np.minimum(raw_lengths, int(width)).astype(np.int64, copy=False)
    if int(lengths.sum()) <= 0:
        return padded, lengths

    fixed_raw_length = int(raw_lengths[0])
    if fixed_raw_length > 0 and bool(np.all(raw_lengths == fixed_raw_length)):
        source_start = int(starts[0])
        source_end = source_start + row_count * fixed_raw_length
        flat_values = values[source_start:source_end]
        if flat_values.shape[0] == row_count * fixed_raw_length:
            use_len = min(fixed_raw_length, int(width))
            padded[:, :use_len] = flat_values.reshape(row_count, fixed_raw_length)[
                :, :use_len
            ]
            return padded, lengths

    for row_index, use_len in enumerate(lengths):
        if use_len <= 0:
            continue
        start = int(starts[row_index])
        padded[row_index, :use_len] = values[start : start + int(use_len)]
    return padded, lengths


def _sequence_event_signatures(
    tokens: NDArray[np.int64],
    lengths: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[Any]]:
    """Return per-row capped lengths, active-event mask and exact event signatures."""
    batch_size, feature_count, max_len = tokens.shape
    raw_lengths = np.minimum(np.maximum(lengths, 0), max_len).astype(np.int64, copy=False)
    positions = np.arange(max_len)
    length_mask = positions[None, :] < raw_lengths[:, None]
    active = length_mask & np.any(tokens > 0, axis=1)
    event_dtype = np.dtype([(f"f{index}", np.int64) for index in range(feature_count)])
    events = np.moveaxis(tokens, 1, 2).copy()
    signatures = events.view(event_dtype).reshape(batch_size, max_len)
    return raw_lengths, active, signatures


def _event_signature_hashes(signatures: NDArray[Any]) -> np.ndarray:
    """64-bit mix of each event's features; collisions are re-verified exactly."""
    events = signatures.view(np.uint64).reshape(*signatures.shape, -1)
    weights = (
        np.arange(events.shape[-1], dtype=np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
        ^ np.uint64(0x9E3779B97F4A7C15)
    )
    hashes = np.bitwise_xor.reduce(events * weights[None, None, :], axis=-1)
    hashes = (hashes ^ (hashes >> np.uint64(33))) * np.uint64(0xFF51AFD7ED558CCD)
    hashes ^= hashes >> np.uint64(29)
    return hashes


def _active_event_groups(
    active: NDArray[np.bool_],
    signatures: NDArray[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group active events by exact signature; return (row_ids, order, group_ends).

    ``order`` sorts active events by (row, signature hash, original position) so
    each exact-signature block ends at the rightmost occurrence; hash collisions
    are split again with an exact structured sort.
    """
    batch_size, max_len = signatures.shape
    flat_active = active.reshape(-1)
    row_ids = np.repeat(np.arange(batch_size), max_len)[flat_active]
    hashes = _event_signature_hashes(signatures).reshape(-1)[flat_active]
    row_bits = max(1, (batch_size - 1).bit_length())
    hash_mask = (np.uint64(1) << (64 - row_bits)) - 1
    combined = (row_ids.astype(np.uint64) << (64 - row_bits)) | (hashes & hash_mask)
    order = np.argsort(combined, kind="stable")
    if len(order) == 0:
        return row_ids, order, np.empty(0, dtype=bool)
    sorted_combined = combined[order]
    boundary = np.empty(len(order), dtype=bool)
    boundary[0] = True
    boundary[1:] = sorted_combined[1:] != sorted_combined[:-1]
    sig_flat = signatures.reshape(-1)[flat_active]
    sig_sorted = sig_flat[order]
    group_starts = np.flatnonzero(boundary)
    group_ids = np.cumsum(boundary) - 1
    first_sigs = sig_sorted[group_starts[group_ids]]
    mixed = sig_sorted != first_sigs
    if mixed.any():
        # hash collision: re-sort every member of affected groups exactly
        mixed_groups = np.unique(group_ids[mixed])
        members = np.flatnonzero(np.isin(group_ids, mixed_groups))
        member_rows = row_ids[order[members]]
        member_positions = np.flatnonzero(flat_active)[order[members]] % max_len
        member_sigs = sig_flat[order[members]]
        composite = np.empty(
            len(members),
            dtype=np.dtype(
                [("row", np.int64), ("sig", member_sigs.dtype), ("pos", np.int64)]
            ),
        )
        composite["row"] = member_rows
        composite["sig"] = member_sigs
        composite["pos"] = member_positions
        order = order.copy()
        order[members] = order[members[np.argsort(composite, kind="stable")]]
        sig_sorted = sig_flat[order]
        boundary[1:] = (sorted_combined[1:] != sorted_combined[:-1]) | (
            sig_sorted[1:] != sig_sorted[:-1]
        )
    group_ends = np.roll(boundary, -1)
    group_ends[-1] = True
    return row_ids, order, group_ends


def build_pcvr_column_plan(
    layout: PCVRSchemaLayout,
    parquet_schema_names: list[str],
) -> PCVRColumnPlan:
    required_column_names = layout.required_column_names(parquet_schema_names)
    column_indices = {name: index for index, name in enumerate(required_column_names)}

    user_int = _build_int_column_plan(
        layout.user_int_cols,
        column_indices=column_indices,
        prefix="user_int_feats",
    )
    item_int = _build_int_column_plan(
        layout.item_int_cols,
        column_indices=column_indices,
        prefix="item_int_feats",
    )

    user_dense: list[DenseColumnPlan] = []
    output_offset = 0
    for fid, dim in layout.user_dense_cols:
        user_dense.append(
            DenseColumnPlan(
                column_index=column_indices[f"user_dense_feats_{fid}"],
                dim=dim,
                output_offset=output_offset,
            )
        )
        output_offset += dim

    sequences: dict[str, SequenceColumnPlan] = {}
    for domain in layout.seq_domains:
        sequence_layout = layout.sequences[domain]
        side_columns = tuple(
            SequenceSideColumnPlan(
                column_index=column_indices[f"{sequence_layout.prefix}_{fid}"],
                slot=slot,
                vocab_size=sequence_layout.vocab_sizes[fid],
            )
            for slot, fid in enumerate(sequence_layout.sideinfo_fids)
        )
        timestamp_column_index = None
        if sequence_layout.timestamp_fid is not None:
            timestamp_column_index = column_indices[
                f"{sequence_layout.prefix}_{sequence_layout.timestamp_fid}"
            ]
        sequences[domain] = SequenceColumnPlan(
            domain=domain,
            max_len=sequence_layout.max_len,
            side_columns=side_columns,
            timestamp_column_index=timestamp_column_index,
        )

    return PCVRColumnPlan(
        required_column_names=required_column_names,
        column_indices=column_indices,
        user_int=user_int,
        item_int=item_int,
        user_dense=tuple(user_dense),
        sequences=sequences,
    )


def _build_int_column_plan(
    columns: tuple[tuple[int, int, int], ...],
    *,
    column_indices: dict[str, int],
    prefix: str,
) -> tuple[IntColumnPlan, ...]:
    plan: list[IntColumnPlan] = []
    output_offset = 0
    for fid, vocab_size, dim in columns:
        plan.append(
            IntColumnPlan(
                column_index=column_indices[f"{prefix}_{fid}"],
                dim=dim,
                output_offset=output_offset,
                vocab_size=vocab_size,
            )
        )
        output_offset += dim
    return tuple(plan)


class PCVRRecordBatchConverter:
    def __init__(
        self,
        *,
        layout: PCVRSchemaLayout,
        column_plan: PCVRColumnPlan,
        batch_size: int,
        clip_vocab: bool,
        is_training: bool,
        strict_time_filter: bool,
    ) -> None:
        self.layout = layout
        self.column_plan = column_plan
        self.batch_size = batch_size
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self.strict_time_filter = strict_time_filter
        self.oob_stats: dict[tuple[str, int], dict[str, int]] = {}

        self.user_int_buffer = np.zeros(
            (batch_size, layout.user_int_schema.total_dim), dtype=np.int64
        )
        self.item_int_buffer = np.zeros(
            (batch_size, layout.item_int_schema.total_dim), dtype=np.int64
        )
        self.user_dense_buffer = np.zeros(
            (batch_size, layout.user_dense_schema.total_dim), dtype=np.float32
        )
        self.user_int_missing_buffer = np.ones(
            (batch_size, layout.user_int_schema.total_dim), dtype=np.bool_
        )
        self.item_int_missing_buffer = np.ones(
            (batch_size, layout.item_int_schema.total_dim), dtype=np.bool_
        )
        self.user_dense_missing_buffer = np.ones(
            (batch_size, layout.user_dense_schema.total_dim), dtype=np.bool_
        )
        self.sequence_buffers = {
            domain: np.zeros(
                (
                    batch_size,
                    len(layout.sideinfo_fids[domain]),
                    layout.seq_maxlen[domain],
                ),
                dtype=np.int64,
            )
            for domain in layout.seq_domains
        }
        self.sequence_lengths = {
            domain: np.zeros(batch_size, dtype=np.int64) for domain in layout.seq_domains
        }
        self.sequence_time_buckets = {
            domain: np.zeros((batch_size, layout.seq_maxlen[domain]), dtype=np.int64)
            for domain in layout.seq_domains
        }
        self.sequence_stats = {
            domain: np.zeros((batch_size, SEQUENCE_STATS_DIM), dtype=np.float32)
            for domain in layout.seq_domains
        }

    def pad_int_column(
        self,
        arrow_col: pa.ListArray,
        width: int,
        row_count: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        padded, lengths = pad_list_offsets_values(
            arrow_col.offsets.to_numpy(),
            arrow_col.values.to_numpy(),
            row_count=row_count,
            width=width,
            dtype=np.int64,
        )
        padded[padded <= 0] = 0
        return padded, lengths

    def pad_float_column(
        self,
        arrow_col: pa.ListArray,
        width: int,
        row_count: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        padded, lengths = pad_list_offsets_values(
            arrow_col.offsets.to_numpy(),
            arrow_col.values.to_numpy(),
            row_count=row_count,
            width=width,
            dtype=np.float32,
        )
        positions = np.arange(width).reshape(1, -1) if width > 0 else np.zeros((1, 0), dtype=np.int64)
        present = positions < lengths.reshape(-1, 1)
        finite = np.isfinite(padded)
        missing = ~(present & finite)
        padded[~finite] = 0.0
        return padded, missing

    def convert(self, batch: pa.RecordBatch) -> dict[str, Any]:
        row_count = batch.num_rows
        timestamps = self._timestamps(batch)
        result = self._base_result(batch, row_count, timestamps)
        self._fill_int_features(
            batch,
            row_count,
            plan=self.column_plan.user_int,
            buffer=self.user_int_buffer[:row_count],
            missing_buffer=self.user_int_missing_buffer[:row_count],
            group="user_int",
        )
        self._fill_int_features(
            batch,
            row_count,
            plan=self.column_plan.item_int,
            buffer=self.item_int_buffer[:row_count],
            missing_buffer=self.item_int_missing_buffer[:row_count],
            group="item_int",
        )
        self._fill_dense_features(batch, row_count)

        result["user_int_feats"] = torch.from_numpy(self.user_int_buffer[:row_count].copy())
        result["item_int_feats"] = torch.from_numpy(self.item_int_buffer[:row_count].copy())
        result["user_dense_feats"] = torch.from_numpy(
            self.user_dense_buffer[:row_count].copy()
        )
        result["user_int_missing_mask"] = torch.from_numpy(self.user_int_missing_buffer[:row_count].copy())
        result["item_int_missing_mask"] = torch.from_numpy(self.item_int_missing_buffer[:row_count].copy())
        result["user_dense_missing_mask"] = torch.from_numpy(self.user_dense_missing_buffer[:row_count].copy())
        result["item_dense_missing_mask"] = torch.zeros(row_count, 0, dtype=torch.bool)
        self._add_sequence_features(batch, row_count, timestamps, result)
        return result

    def dump_oob_stats(self, path: str | None = None) -> None:
        if not self.oob_stats:
            logger.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, column_index), stats in sorted(self.oob_stats.items()):
            direction = "TOO_HIGH" if stats["min_oob"] >= stats["vocab"] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={column_index}: vocab={stats['vocab']}, "
                f"oob_count={stats['count']}, range=[{stats['min_oob']}, {stats['max']}], "
                f"{direction}"
            )
        message = "\n".join(lines)
        if path:
            with Path(path).open("w") as file:
                file.write(message + "\n")
            logger.info("OOB stats written to {}", path)
        else:
            logger.info(message)

    def _timestamps(self, batch: pa.RecordBatch) -> NDArray[np.int64]:
        return batch.column(self.column_plan.column_indices["timestamp"]).to_numpy().astype(np.int64)

    def _base_result(
        self,
        batch: pa.RecordBatch,
        row_count: int,
        timestamps: NDArray[np.int64],
    ) -> dict[str, Any]:
        labels = self._labels(batch, row_count)
        user_ids = batch.column(self.column_plan.column_indices["user_id"]).to_pylist()
        return {
            "item_dense_feats": torch.zeros(row_count, 0, dtype=torch.float32),
            "label": torch.from_numpy(labels),
            "timestamp": torch.from_numpy(timestamps),
            "user_id": user_ids,
            "_seq_domains": self.layout.seq_domains,
        }

    def _labels(self, batch: pa.RecordBatch, row_count: int) -> NDArray[np.int64]:
        if not self.is_training:
            return np.zeros(row_count, dtype=np.int64)
        return (
            batch.column(self.column_plan.column_indices["label_type"])
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
            == 2
        ).astype(np.int64)

    def _fill_int_features(
        self,
        batch: pa.RecordBatch,
        row_count: int,
        *,
        plan: tuple[IntColumnPlan, ...],
        buffer: NDArray[np.int64],
        missing_buffer: NDArray[np.bool_],
        group: str,
    ) -> None:
        buffer[:] = 0
        missing_buffer[:] = True
        for feature in plan:
            column = batch.column(feature.column_index)
            if feature.dim == 1:
                values, missing = self._scalar_int_values_and_missing(column)
                if feature.vocab_size > 0:
                    self.record_oob(group, feature.column_index, values, feature.vocab_size)
                else:
                    values[:] = 0
                buffer[:, feature.output_offset] = values
                missing_buffer[:, feature.output_offset] = missing | (values <= 0)
                continue

            padded, _lengths = self.pad_int_column(column, feature.dim, row_count)
            if feature.vocab_size > 0:
                self.record_oob(group, feature.column_index, padded, feature.vocab_size)
            else:
                padded[:] = 0
            buffer[:, feature.output_offset : feature.output_offset + feature.dim] = padded
            missing_buffer[:, feature.output_offset : feature.output_offset + feature.dim] = padded <= 0

    def _scalar_int_values_and_missing(self, column: pa.Array) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        values = column.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        missing = column.is_null().to_numpy(zero_copy_only=False).astype(np.bool_) | (values <= 0)
        values[values <= 0] = 0
        return values, missing

    def _fill_dense_features(self, batch: pa.RecordBatch, row_count: int) -> None:
        buffer = self.user_dense_buffer[:row_count]
        missing_buffer = self.user_dense_missing_buffer[:row_count]
        buffer[:] = 0
        missing_buffer[:] = True
        for feature in self.column_plan.user_dense:
            padded, missing = self.pad_float_column(batch.column(feature.column_index), feature.dim, row_count)
            buffer[:, feature.output_offset : feature.output_offset + feature.dim] = padded
            missing_buffer[:, feature.output_offset : feature.output_offset + feature.dim] = missing

    def _add_sequence_features(
        self,
        batch: pa.RecordBatch,
        row_count: int,
        timestamps: NDArray[np.int64],
        result: dict[str, Any],
    ) -> None:
        for domain in self.layout.seq_domains:
            sequence_plan = self.column_plan.sequences[domain]
            tokens = self.sequence_buffers[domain][:row_count]
            lengths = self.sequence_lengths[domain][:row_count]
            time_buckets = self.sequence_time_buckets[domain][:row_count]
            stats = self.sequence_stats[domain][:row_count]
            tokens[:] = 0
            lengths[:] = 0
            time_buckets[:] = 0
            stats[:] = 0.0

            side_columns = self._sequence_side_arrays(batch, sequence_plan)
            timestamps_padded = np.zeros((row_count, sequence_plan.max_len), dtype=np.int64)
            if self.strict_time_filter and sequence_plan.timestamp_column_index is not None:
                self._fill_strict_sequence(
                    batch=batch,
                    row_count=row_count,
                    timestamps=timestamps,
                    sequence_plan=sequence_plan,
                    side_columns=side_columns,
                    tokens=tokens,
                    lengths=lengths,
                    timestamps_padded=timestamps_padded,
                )
            else:
                self._fill_sequence(
                    batch=batch,
                    row_count=row_count,
                    sequence_plan=sequence_plan,
                    side_columns=side_columns,
                    tokens=tokens,
                    lengths=lengths,
                    timestamps_padded=timestamps_padded,
                )

            tokens[tokens <= 0] = 0
            self._clip_sequence_vocab(domain, sequence_plan, tokens)
            self._sequence_stats_and_dedup(tokens, lengths, timestamps_padded, stats)
            self._fill_time_buckets(timestamps, timestamps_padded, time_buckets)
            self._fill_sequence_time_stats(lengths, time_buckets, stats)

            result[domain] = torch.from_numpy(tokens.copy())
            result[f"{domain}_len"] = torch.from_numpy(lengths.copy())
            result[f"{domain}_time_bucket"] = torch.from_numpy(time_buckets.copy())
            result[f"{domain}_stats"] = torch.from_numpy(stats.copy())

    def _sequence_side_arrays(
        self,
        batch: pa.RecordBatch,
        sequence_plan: SequenceColumnPlan,
    ) -> tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...]:
        arrays = []
        for side_column in sequence_plan.side_columns:
            column = batch.column(side_column.column_index)
            arrays.append((column.offsets.to_numpy(), column.values.to_numpy(), side_column))
        return tuple(arrays)

    def _fill_sequence(
        self,
        *,
        batch: pa.RecordBatch,
        row_count: int,
        sequence_plan: SequenceColumnPlan,
        side_columns: tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...],
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
    ) -> None:
        for offsets, values, side_column in side_columns:
            padded, feature_lengths = pad_list_offsets_values(
                offsets,
                values,
                row_count=row_count,
                width=sequence_plan.max_len,
                dtype=np.int64,
            )
            tokens[:, side_column.slot, :] = padded
            np.maximum(lengths, feature_lengths, out=lengths)

        if sequence_plan.timestamp_column_index is None:
            return
        timestamp_column = batch.column(sequence_plan.timestamp_column_index)
        timestamps_padded[:, :] = pad_list_offsets_values(
            timestamp_column.offsets.to_numpy(),
            timestamp_column.values.to_numpy(),
            row_count=row_count,
            width=sequence_plan.max_len,
            dtype=np.int64,
        )[0]

    def _fill_strict_sequence(
        self,
        *,
        batch: pa.RecordBatch,
        row_count: int,
        timestamps: NDArray[np.int64],
        sequence_plan: SequenceColumnPlan,
        side_columns: tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...],
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
    ) -> None:
        timestamp_column = batch.column(sequence_plan.timestamp_column_index)
        timestamp_offsets = timestamp_column.offsets.to_numpy()
        timestamp_values = timestamp_column.values.to_numpy()
        for row_index in range(row_count):
            start = int(timestamp_offsets[row_index])
            end = int(timestamp_offsets[row_index + 1])
            if end <= start:
                continue
            row_timestamps = timestamp_values[start:end]
            valid_positions = np.flatnonzero(
                (row_timestamps > 0) & (row_timestamps < timestamps[row_index])
            )
            if valid_positions.size == 0:
                continue
            if valid_positions.size > sequence_plan.max_len:
                valid_positions = valid_positions[-sequence_plan.max_len :]

            valid_length = int(valid_positions.size)
            lengths[row_index] = valid_length
            timestamps_padded[row_index, :valid_length] = row_timestamps[valid_positions]
            self._copy_strict_side_columns(row_index, valid_positions, side_columns, tokens)

    def _copy_strict_side_columns(
        self,
        row_index: int,
        valid_positions: NDArray[np.int64],
        side_columns: tuple[tuple[NDArray[Any], NDArray[Any], SequenceSideColumnPlan], ...],
        tokens: NDArray[np.int64],
    ) -> None:
        for offsets, values, side_column in side_columns:
            start = int(offsets[row_index])
            end = int(offsets[row_index + 1])
            side_len = end - start
            if side_len <= 0:
                continue
            side_positions = valid_positions[valid_positions < side_len]
            if side_positions.size == 0:
                continue
            tokens[row_index, side_column.slot, : side_positions.size] = values[
                start + side_positions
            ]

    def _clip_sequence_vocab(
        self,
        domain: str,
        sequence_plan: SequenceColumnPlan,
        tokens: NDArray[np.int64],
    ) -> None:
        for side_column in sequence_plan.side_columns:
            slice_tokens = tokens[:, side_column.slot, :]
            if side_column.vocab_size > 0:
                self.record_oob(
                    f"seq_{domain}",
                    side_column.column_index,
                    slice_tokens,
                    side_column.vocab_size,
                )
            else:
                slice_tokens[:] = 0

    def _fill_time_buckets(
        self,
        timestamps: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
        time_buckets: NDArray[np.int64],
    ) -> None:
        if timestamps_padded.shape[1] == 0:
            return
        time_diff = np.maximum(timestamps.reshape(-1, 1) - timestamps_padded, 0)
        raw_buckets = np.clip(
            np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
            0,
            len(BUCKET_BOUNDARIES) - 1,
        )
        buckets = raw_buckets.reshape(timestamps_padded.shape) + 1
        buckets[timestamps_padded == 0] = 0
        time_buckets[:] = buckets

    def _sequence_stats_and_dedup(
        self,
        tokens: NDArray[np.int64],
        lengths: NDArray[np.int64],
        timestamps_padded: NDArray[np.int64],
        stats: NDArray[np.float32],
    ) -> None:
        """Fill raw sequence stats and deduplicate events in one grouped pass."""
        batch_size, feature_count, max_len = tokens.shape
        raw_lengths = np.minimum(np.maximum(lengths, 0), max_len).astype(np.int64, copy=False)
        if feature_count <= 0:
            # stats stay zeroed; zero-length event tuples are never kept, so rows longer than 1 get emptied
            for row_index in np.flatnonzero(raw_lengths > 1):
                tokens[row_index].fill(0)
                timestamps_padded[row_index].fill(0)
                lengths[row_index] = 0
            return
        raw_lengths, active, signatures = _sequence_event_signatures(tokens, lengths)
        row_ids, order, group_ends = _active_event_groups(active, signatures)

        active_count = active.sum(axis=1)
        has_events = active_count > 0
        if len(order) > 0:
            unique_count = np.bincount(row_ids[group_ends], minlength=batch_size)
        else:
            unique_count = np.zeros(batch_size, dtype=np.int64)
        nonzero = (tokens > 0) & (np.arange(max_len)[None, :] < raw_lengths[:, None])[:, None, :]
        nonzero_fraction = nonzero.sum(axis=(1, 2)) / np.maximum(raw_lengths * feature_count, 1)
        stats[:, 0] = np.where(has_events, raw_lengths, 0)
        stats[:, 1] = np.where(has_events, active_count, 0)
        stats[:, 2] = np.where(has_events, unique_count, 0)
        stats[:, 3] = np.where(
            has_events, 1.0 - unique_count / np.maximum(active_count, 1), 0.0
        )
        stats[:, 4] = np.where(has_events, nonzero_fraction, 0.0)

        flat_active = active.reshape(-1)
        if not bool(flat_active.any()):
            return
        kept_flat = np.flatnonzero(flat_active)[order[group_ends]]
        kept_rows = kept_flat // max_len
        kept_positions = kept_flat % max_len
        # the reference semantics never touch rows with raw_length <= 1
        dedup_rows = raw_lengths > 1
        row_keep = dedup_rows[kept_rows]
        kept_rows = kept_rows[row_keep]
        kept_positions = kept_positions[row_keep]
        new_lengths = np.bincount(kept_rows, minlength=batch_size)
        changed = (new_lengths != raw_lengths) & dedup_rows
        if not bool(changed.any()):
            return
        order2 = np.lexsort((kept_positions, kept_rows))
        kr = kept_rows[order2]
        kp = kept_positions[order2]
        counts = np.bincount(kr, minlength=batch_size)
        offsets = np.zeros(batch_size + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        segment_positions = np.arange(int(kr.size)) - np.repeat(offsets[:-1], counts)
        values = tokens[kr[:, None], np.arange(feature_count)[None, :], kp[:, None]]
        timestamp_values = timestamps_padded[kr, kp]
        changed_rows = np.flatnonzero(changed)
        for row_index in changed_rows:
            tokens[row_index].fill(0)
            timestamps_padded[row_index].fill(0)
        tokens[kr[:, None], np.arange(feature_count)[None, :], segment_positions[:, None]] = values
        timestamps_padded[kr, segment_positions] = timestamp_values
        lengths[changed_rows] = new_lengths[changed_rows]

    def _fill_sequence_time_stats(
        self,
        lengths: NDArray[np.int64],
        time_buckets: NDArray[np.int64],
        stats: NDArray[np.float32],
    ) -> None:
        max_len = time_buckets.shape[1]
        for row_index, length_value in enumerate(lengths):
            length = min(max(int(length_value), 0), max_len)
            if length <= 0:
                continue
            stats[row_index, 1] = float(length)
            stats[row_index, 5] = float(time_buckets[row_index, length - 1])

    def record_oob(
        self,
        group: str,
        column_index: int,
        values: NDArray[np.int64],
        vocab_size: int,
    ) -> None:
        oob_mask = values >= vocab_size
        if not oob_mask.any():
            return
        oob_values = values[oob_mask]
        count = int(oob_mask.sum())
        max_value = int(oob_values.max())
        min_oob = int(oob_values.min())
        key = (group, column_index)
        if key in self.oob_stats:
            stats = self.oob_stats[key]
            stats["count"] += count
            stats["max"] = max(stats["max"], max_value)
            stats["min_oob"] = min(stats["min_oob"], min_oob)
        else:
            self.oob_stats[key] = {
                "count": count,
                "max": max_value,
                "min_oob": min_oob,
                "vocab": vocab_size,
            }
        if self.clip_vocab:
            values[oob_mask] = 0
            return
        raise ValueError(
            f"{group} col_idx={column_index}: {count} values out of range "
            f"[0, {vocab_size}), actual=[{min_oob}, {max_value}]. "
            "Use clip_vocab=True to clip or fix schema.json"
        )
