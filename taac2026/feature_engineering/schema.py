from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_len(value: Any) -> int:
    return 0 if value is None else len(value)


def _new_array_stats() -> dict[str, float]:
    return {"count": 0.0, "sum": 0.0, "min": float("inf"), "max": 0.0}


def _update_array_stats(target: dict[str, float], values: Any) -> None:
    if values is None:
        return
    array = np.asarray(values)
    target["count"] += 1
    target["sum"] += float(array.size)
    target["min"] = min(target["min"], float(array.size))
    target["max"] = max(target["max"], float(array.size))


def _finalize_length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}
    values = np.asarray(lengths, dtype=np.float32)
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _finalize_array_stats(stats: dict[str, float]) -> dict[str, float]:
    if stats["count"] == 0:
        return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": stats["count"],
        "mean": stats["sum"] / stats["count"],
        "min": stats["min"],
        "max": stats["max"],
    }


def _new_feature_profile() -> dict[str, Any]:
    return {
        "occurrences": 0,
        "value_types": Counter(),
        "int_value_present": 0,
        "float_value_present": 0,
        "int_array_length": _new_array_stats(),
        "float_array_length": _new_array_stats(),
        "timestamp_like_count": 0,
    }


def _update_feature_profile(profile: dict[str, Any], feature: dict[str, Any], allow_timestamp_heuristic: bool) -> None:
    profile["occurrences"] += 1
    profile["value_types"][str(feature["feature_value_type"])] += 1

    int_value = feature.get("int_value")
    if int_value is not None and not pd.isna(int_value):
        profile["int_value_present"] += 1

    float_value = feature.get("float_value")
    if float_value is not None and not pd.isna(float_value):
        profile["float_value_present"] += 1

    int_array = feature.get("int_array")
    float_array = feature.get("float_array")
    _update_array_stats(profile["int_array_length"], int_array)
    _update_array_stats(profile["float_array_length"], float_array)

    if allow_timestamp_heuristic and int_array is not None:
        array = np.asarray(int_array)
        if array.size > 0 and float(np.median(array)) > 1_000_000_000:
            profile["timestamp_like_count"] += 1


def _finalize_feature_dictionary(feature_profiles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    finalized: dict[str, Any] = {}
    for feature_id, profile in sorted(feature_profiles.items()):
        finalized[str(feature_id)] = {
            "occurrences": profile["occurrences"],
            "value_type_counts": dict(profile["value_types"]),
            "int_value_present": profile["int_value_present"],
            "float_value_present": profile["float_value_present"],
            "int_array_length": _finalize_array_stats(profile["int_array_length"]),
            "float_array_length": _finalize_array_stats(profile["float_array_length"]),
            "timestamp_like_count": profile["timestamp_like_count"],
        }
    return finalized


def _summarize_feature_collection(rows: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_ids = Counter()
    value_types = Counter()
    int_value_present = 0
    float_value_present = 0
    int_array_stats = _new_array_stats()
    float_array_stats = _new_array_stats()
    feature_count_per_row: list[int] = []
    feature_profiles: dict[int, dict[str, Any]] = {}

    for features in rows:
        feature_list = features if features is not None else []
        feature_count_per_row.append(len(feature_list))
        for feature in feature_list:
            feature_id = int(feature["feature_id"])
            feature_ids[feature_id] += 1
            value_types[str(feature["feature_value_type"])] += 1

            int_value = feature.get("int_value")
            if int_value is not None and not pd.isna(int_value):
                int_value_present += 1

            float_value = feature.get("float_value")
            if float_value is not None and not pd.isna(float_value):
                float_value_present += 1

            _update_array_stats(int_array_stats, feature.get("int_array"))
            _update_array_stats(float_array_stats, feature.get("float_array"))

            profile = feature_profiles.setdefault(feature_id, _new_feature_profile())
            _update_feature_profile(profile, feature, allow_timestamp_heuristic=False)

    summary = {
        "rows": _finalize_length_stats(feature_count_per_row),
        "unique_feature_ids": len(feature_ids),
        "top_feature_ids": feature_ids.most_common(20),
        "value_type_counts": dict(value_types),
        "int_value_present": int_value_present,
        "float_value_present": float_value_present,
        "int_array_length": _finalize_array_stats(int_array_stats),
        "float_array_length": _finalize_array_stats(float_array_stats),
    }
    return summary, _finalize_feature_dictionary(feature_profiles)


def _summarize_sequence_groups(rows: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    per_group: dict[str, dict[str, Any]] = {}
    feature_profiles_by_group: dict[str, dict[int, dict[str, Any]]] = {}

    for seq_feature in rows:
        for group_name, features in seq_feature.items():
            group = per_group.setdefault(
                group_name,
                {
                    "row_lengths": [],
                    "feature_count_per_row": [],
                    "feature_ids": Counter(),
                    "value_types": Counter(),
                    "timestamp_feature_candidates": Counter(),
                },
            )
            feature_profiles = feature_profiles_by_group.setdefault(group_name, {})

            feature_list = features if features is not None else []
            group["feature_count_per_row"].append(len(feature_list))
            group_lengths = []

            for feature in feature_list:
                feature_id = int(feature["feature_id"])
                group["feature_ids"][feature_id] += 1
                group["value_types"][str(feature["feature_value_type"])] += 1
                values = feature.get("int_array")
                if values is not None:
                    array = np.asarray(values)
                    group_lengths.append(int(array.size))
                    if array.size > 0 and float(np.median(array)) > 1_000_000_000:
                        group["timestamp_feature_candidates"][feature_id] += 1

                profile = feature_profiles.setdefault(feature_id, _new_feature_profile())
                _update_feature_profile(profile, feature, allow_timestamp_heuristic=True)

            group["row_lengths"].append(int(min(group_lengths)) if group_lengths else 0)

    finalized_summary = {}
    finalized_dictionary = {}
    for group_name, group in per_group.items():
        finalized_summary[group_name] = {
            "row_lengths": _finalize_length_stats(group["row_lengths"]),
            "feature_count_per_row": _finalize_length_stats(group["feature_count_per_row"]),
            "unique_feature_ids": len(group["feature_ids"]),
            "top_feature_ids": group["feature_ids"].most_common(20),
            "value_type_counts": dict(group["value_types"]),
            "timestamp_feature_candidates": group["timestamp_feature_candidates"].most_common(10),
        }
        finalized_dictionary[group_name] = _finalize_feature_dictionary(feature_profiles_by_group[group_name])
    return finalized_summary, finalized_dictionary


def build_schema_artifacts(dataset_path: str | Path) -> dict[str, Any]:
    dataframe = pd.read_parquet(Path(dataset_path))
    timestamp_values = dataframe["timestamp"].to_numpy(dtype=np.int64)

    label_lengths = []
    action_type_counts = Counter()
    for labels in dataframe["label"]:
        label_lengths.append(_safe_len(labels))
        for label in labels:
            action_type_counts[int(label["action_type"])] += 1

    user_summary, user_feature_dictionary = _summarize_feature_collection(dataframe["user_feature"])
    item_summary, item_feature_dictionary = _summarize_feature_collection(dataframe["item_feature"])
    sequence_summary, sequence_feature_dictionary = _summarize_sequence_groups(dataframe["seq_feature"])

    schema_summary = {
        "dataset": {
            "path": str(dataset_path),
            "rows": int(dataframe.shape[0]),
            "columns": dataframe.columns.tolist(),
            "timestamp": {
                "min": int(timestamp_values.min()),
                "max": int(timestamp_values.max()),
            },
        },
        "labels": {
            "length": _finalize_length_stats(label_lengths),
            "action_type_counts": dict(action_type_counts),
        },
        "user_feature": user_summary,
        "item_feature": item_summary,
        "sequence": sequence_summary,
    }

    feature_dictionary = {
        "dataset_path": str(dataset_path),
        "user_feature": user_feature_dictionary,
        "item_feature": item_feature_dictionary,
        "sequence": sequence_feature_dictionary,
    }
    return {
        "schema_summary": schema_summary,
        "feature_dictionary": feature_dictionary,
    }


def print_human_summary(summary: dict[str, Any]) -> None:
    dataset = summary["dataset"]
    print(f"rows={dataset['rows']} columns={dataset['columns']}")
    print(f"timestamp_range=[{dataset['timestamp']['min']}, {dataset['timestamp']['max']}]")
    print(f"label_action_type_counts={summary['labels']['action_type_counts']}")
    print(f"user_unique_feature_ids={summary['user_feature']['unique_feature_ids']}")
    print(f"item_unique_feature_ids={summary['item_feature']['unique_feature_ids']}")
    for group_name, group_summary in summary["sequence"].items():
        print(
            f"sequence_group={group_name} mean_length={group_summary['row_lengths']['mean']:.2f} max_length={group_summary['row_lengths']['max']:.0f} timestamp_candidates={group_summary['timestamp_feature_candidates']}"
        )


__all__ = ["build_schema_artifacts", "print_human_summary"]