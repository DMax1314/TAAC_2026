"""Dataset EDA CLI for labeled test/sample data and online formal data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from taac2026.infrastructure.pcvr.protocol import resolve_schema_path


TEST_DEFAULT_OUTPUT = Path("outputs/reports/dataset_eda.json")
TEST_DEFAULT_CHART_DIR = Path("docs/assets/figures/eda")
ONLINE_DEFAULT_OUTPUT = Path("outputs/reports/online_dataset_eda.json")
ONLINE_DEFAULT_CHART_DIR = Path("outputs/reports/online_dataset_eda_charts")
TEST_ONLY_CHARTS = ("label_distribution", "feature_auc", "null_rate_by_label")


class DatasetRole(str, Enum):
    AUTO = "auto"
    TEST = "test"
    ONLINE = "online"


@dataclass(slots=True)
class SequenceDomainLayout:
    name: str
    prefix: str
    ts_column: str | None
    sideinfo_columns: tuple[str, ...]

    @property
    def all_columns(self) -> tuple[str, ...]:
        columns = list(self.sideinfo_columns)
        if self.ts_column is not None:
            columns.append(self.ts_column)
        return tuple(columns)

    @property
    def length_column(self) -> str | None:
        if self.ts_column is not None:
            return self.ts_column
        if self.sideinfo_columns:
            return self.sideinfo_columns[0]
        return None

    @property
    def repeat_column(self) -> str | None:
        if self.sideinfo_columns:
            return self.sideinfo_columns[0]
        return self.ts_column


@dataclass(slots=True)
class SchemaLayout:
    user_int_columns: tuple[str, ...]
    item_int_columns: tuple[str, ...]
    user_dense_columns: tuple[str, ...]
    sequence_domains: tuple[SequenceDomainLayout, ...]

    @classmethod
    def from_path(cls, path: Path) -> SchemaLayout:
        raw = json.loads(path.read_text(encoding="utf-8"))
        user_int_columns = tuple(f"user_int_feats_{fid}" for fid, _vocab_size, _dim in raw["user_int"])
        item_int_columns = tuple(f"item_int_feats_{fid}" for fid, _vocab_size, _dim in raw["item_int"])
        user_dense_columns = tuple(f"user_dense_feats_{fid}" for fid, _dim in raw["user_dense"])
        sequence_domains: list[SequenceDomainLayout] = []
        for name, config in sorted(raw["seq"].items()):
            prefix = str(config["prefix"])
            ts_fid = config.get("ts_fid")
            ts_column = f"{prefix}_{ts_fid}" if ts_fid is not None else None
            sideinfo_columns = tuple(
                f"{prefix}_{fid}"
                for fid, _vocab_size in config["features"]
                if fid != ts_fid
            )
            sequence_domains.append(
                SequenceDomainLayout(
                    name=name,
                    prefix=prefix,
                    ts_column=ts_column,
                    sideinfo_columns=sideinfo_columns,
                )
            )
        return cls(
            user_int_columns=user_int_columns,
            item_int_columns=item_int_columns,
            user_dense_columns=user_dense_columns,
            sequence_domains=tuple(sequence_domains),
        )

    @property
    def sequence_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for domain in self.sequence_domains:
            columns.extend(domain.all_columns)
        return tuple(columns)

    @property
    def sparse_columns(self) -> tuple[str, ...]:
        return self.user_int_columns + self.item_int_columns

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.user_int_columns + self.user_dense_columns + self.item_int_columns + self.sequence_columns

    @property
    def primary_user_id_column(self) -> str | None:
        if not self.user_int_columns:
            return None
        return self.user_int_columns[0]

    @property
    def group_by_column(self) -> dict[str, str]:
        result: dict[str, str] = {}
        result.update({column: "user_int" for column in self.user_int_columns})
        result.update({column: "user_dense" for column in self.user_dense_columns})
        result.update({column: "item_int" for column in self.item_int_columns})
        for domain in self.sequence_domains:
            for column in domain.all_columns:
                result[column] = domain.name
        return result


@dataclass(slots=True)
class LoadedDataset:
    dataset_path: Path
    files: tuple[Path, ...]
    available_columns: tuple[str, ...]
    total_rows: int
    table: pa.Table


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute EDA statistics for TAAC parquet datasets")
    parser.add_argument("--dataset", "--dataset-path", dest="dataset_path", required=True)
    parser.add_argument("--schema-path", default=None)
    parser.add_argument(
        "--dataset-role",
        choices=[role.value for role in DatasetRole],
        default=DatasetRole.AUTO.value,
        help="test=sample/offline labeled dataset, online=formal unlabeled dataset, auto=detect from label columns",
    )
    parser.add_argument("--output", default=None, help="summary JSON path")
    parser.add_argument("--json-path", default=None, help="alias of --output")
    parser.add_argument("--chart-dir", default=None, help="directory for generated ECharts JSON")
    parser.add_argument("--max-rows", type=int, default=None, help="limit scanned rows for large datasets")
    parser.add_argument("--no-charts", action="store_true", help="skip writing ECharts JSON files")
    args = parser.parse_args(argv)
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.output and args.json_path and args.output != args.json_path:
        parser.error("--output and --json-path must match when both are provided")
    return args


def _list_parquet_files(dataset_path: Path) -> tuple[Path, ...]:
    if dataset_path.is_dir():
        files = tuple(sorted(dataset_path.glob("*.parquet")))
    else:
        files = (dataset_path,)
    if not files:
        raise FileNotFoundError(f"No .parquet files found at {dataset_path}")
    return files


def _collect_available_columns(files: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(pq.ParquetFile(files[0]).schema_arrow.names)


def _load_dataset(dataset_path: Path, *, columns: tuple[str, ...], max_rows: int | None) -> LoadedDataset:
    files = _list_parquet_files(dataset_path)
    available_columns = _collect_available_columns(files)
    selected_columns = [column for column in columns if column in available_columns]
    total_rows = 0
    batches: list[pa.RecordBatch] = []
    remaining = max_rows

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        total_rows += parquet_file.metadata.num_rows
        for batch in parquet_file.iter_batches(batch_size=8192, columns=selected_columns):
            if remaining is not None:
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
                remaining -= batch.num_rows
            batches.append(batch)
        if remaining == 0:
            break

    if batches:
        table = pa.Table.from_batches(batches)
    else:
        table = pa.table({column: [] for column in selected_columns})
    return LoadedDataset(
        dataset_path=dataset_path,
        files=files,
        available_columns=available_columns,
        total_rows=total_rows,
        table=table,
    )


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if _normalize_scalar(item) is not None]
    scalar = _normalize_scalar(value)
    return [] if scalar is None else [scalar]


def _is_missing(value: Any) -> bool:
    if isinstance(value, list):
        return len(_normalize_list(value)) == 0
    return _normalize_scalar(value) is None


def _sparse_tokens(value: Any) -> tuple[Any, ...]:
    tokens: list[Any] = []
    for item in _normalize_list(value):
        if isinstance(item, (int, float)) and item <= 0:
            continue
        tokens.append(item)
    return tuple(tokens)


def _hashable_value(value: Any) -> Any:
    if isinstance(value, list):
        tokens = _sparse_tokens(value)
        return tokens or None
    scalar = _normalize_scalar(value)
    if isinstance(scalar, (int, float)) and scalar <= 0:
        return None
    return scalar


def _column_null_rate(values: list[Any]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if _is_missing(value)) / float(len(values))


def _quantile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _detect_label_columns(columns: set[str]) -> tuple[str, ...]:
    return tuple(name for name in ("label_type", "label_action_type") if name in columns)


def resolve_dataset_role(explicit_role: str, columns: set[str]) -> tuple[DatasetRole, tuple[str, ...]]:
    label_columns = _detect_label_columns(columns)
    if explicit_role == DatasetRole.TEST.value:
        if not label_columns:
            raise ValueError("test dataset role requires label_type or label_action_type columns")
        return DatasetRole.TEST, label_columns
    if explicit_role == DatasetRole.ONLINE.value:
        return DatasetRole.ONLINE, label_columns
    if label_columns:
        return DatasetRole.TEST, label_columns
    return DatasetRole.ONLINE, label_columns


def _resolve_output_paths(
    *,
    role: DatasetRole,
    output_arg: str | None,
    json_path_arg: str | None,
    chart_dir_arg: str | None,
    no_charts: bool,
) -> tuple[Path, Path | None]:
    output_path = Path(output_arg or json_path_arg or (TEST_DEFAULT_OUTPUT if role is DatasetRole.TEST else ONLINE_DEFAULT_OUTPUT))
    if no_charts:
        return output_path, None
    default_chart_dir = TEST_DEFAULT_CHART_DIR if role is DatasetRole.TEST else ONLINE_DEFAULT_CHART_DIR
    chart_dir = Path(chart_dir_arg) if chart_dir_arg else default_chart_dir
    return output_path, chart_dir


def _chart_title(text: str, subtitle: str) -> dict[str, str]:
    return {"text": text, "subtext": subtitle}


def _base_subtitle(role: DatasetRole, row_count: int, total_rows: int) -> str:
    if row_count == total_rows:
        return f"{role.value} dataset, {row_count} rows"
    return f"{role.value} dataset, scanned {row_count}/{total_rows} rows"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _series_bar(names: list[str], values: list[float], *, title: str, subtitle: str, series_name: str, horizontal: bool = False) -> dict[str, Any]:
    x_axis = {"type": "category", "data": names, "axisLabel": {"rotate": 35}} if not horizontal else {"type": "value"}
    y_axis = {"type": "value"} if not horizontal else {"type": "category", "data": names}
    series_data: list[Any] = values
    return {
        "title": _chart_title(title, subtitle),
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 100 if horizontal else 60, "right": 30, "top": 70, "bottom": 90 if not horizontal else 40},
        "xAxis": x_axis,
        "yAxis": y_axis,
        "series": [{"name": series_name, "type": "bar", "data": series_data, "itemStyle": {"borderRadius": 6}}],
    }


def _column_layout_chart(layout_counts: dict[str, int], domain_counts: dict[str, int], subtitle: str) -> dict[str, Any]:
    return {
        "title": _chart_title("列布局概览", subtitle),
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "grid": {"left": 50, "right": 30, "top": 70, "bottom": 100},
        "xAxis": {"type": "category", "data": ["scalar", "user_int", "user_dense", "item_int", "sequence"]},
        "yAxis": {"type": "value", "name": "columns"},
        "series": [
            {
                "name": "column_count",
                "type": "bar",
                "data": [
                    layout_counts["scalar"],
                    layout_counts["user_int"],
                    layout_counts["user_dense"],
                    layout_counts["item_int"],
                    layout_counts["sequence"],
                ],
                "itemStyle": {"borderRadius": 6},
            },
            {
                "name": "sequence_domains",
                "type": "pie",
                "radius": ["35%", "55%"],
                "center": ["82%", "35%"],
                "label": {"formatter": "{b}: {c}"},
                "data": [{"name": key, "value": value} for key, value in domain_counts.items()],
            },
        ],
    }


def _label_distribution_chart(counts: Counter[Any], subtitle: str) -> dict[str, Any]:
    name_map = {0: "曝光", 1: "点击", 2: "转化"}
    data = [{"name": name_map.get(key, str(key)), "value": value} for key, value in sorted(counts.items(), key=lambda item: item[0])]
    return {
        "title": _chart_title("行为类型分布", subtitle),
        "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
        "legend": {"bottom": 0},
        "series": [
            {
                "type": "pie",
                "radius": ["35%", "65%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 6, "borderWidth": 2},
                "label": {"formatter": "{b}\n{d}%"},
                "data": data,
            }
        ],
    }


def _heatmap_chart(*, title: str, subtitle: str, x_labels: list[str], y_labels: list[str], data: list[list[Any]], value_name: str) -> dict[str, Any]:
    return {
        "title": _chart_title(title, subtitle),
        "tooltip": {"trigger": "item"},
        "grid": {"left": 80, "right": 30, "top": 70, "bottom": 110},
        "xAxis": {"type": "category", "data": x_labels, "axisLabel": {"rotate": 45}},
        "yAxis": {"type": "category", "data": y_labels},
        "visualMap": {"min": 0, "max": 1, "orient": "horizontal", "left": "center", "bottom": 25, "text": [value_name, "0"]},
        "series": [{"type": "heatmap", "data": data, "itemStyle": {"borderRadius": 2}}],
    }


def _boxplot_chart(domains: list[str], stats: list[list[float]], subtitle: str) -> dict[str, Any]:
    return {
        "title": _chart_title("序列长度分布", subtitle),
        "tooltip": {"trigger": "item"},
        "grid": {"left": 60, "right": 30, "top": 70, "bottom": 60},
        "xAxis": {"type": "category", "data": domains},
        "yAxis": {"type": "value", "name": "length"},
        "series": [{"type": "boxplot", "data": stats}],
    }


def _sequence_summary_chart(domains: list[str], means: list[float], p95s: list[float], empty_rates: list[float], subtitle: str) -> dict[str, Any]:
    return {
        "title": _chart_title("序列长度摘要", subtitle),
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "grid": {"left": 60, "right": 60, "top": 70, "bottom": 90},
        "xAxis": {"type": "category", "data": domains},
        "yAxis": [{"type": "value", "name": "length"}, {"type": "value", "name": "empty_rate", "min": 0, "max": 1}],
        "series": [
            {"name": "mean", "type": "bar", "data": means, "itemStyle": {"borderRadius": 6}},
            {"name": "p95", "type": "bar", "data": p95s, "itemStyle": {"borderRadius": 6}},
            {"name": "empty_rate", "type": "line", "yAxisIndex": 1, "data": empty_rates, "smooth": True},
        ],
    }


def _grouped_bar_chart(*, title: str, subtitle: str, names: list[str], left_name: str, left_values: list[float], right_name: str, right_values: list[float]) -> dict[str, Any]:
    return {
        "title": _chart_title(title, subtitle),
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "grid": {"left": 100, "right": 30, "top": 70, "bottom": 70},
        "xAxis": {"type": "value"},
        "yAxis": {"type": "category", "data": names},
        "series": [
            {"name": left_name, "type": "bar", "data": left_values, "itemStyle": {"borderRadius": 6}},
            {"name": right_name, "type": "bar", "data": right_values, "itemStyle": {"borderRadius": 6}},
        ],
    }


def _scatter_chart(*, title: str, subtitle: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "title": _chart_title(title, subtitle),
        "tooltip": {"trigger": "item"},
        "xAxis": {"type": "value", "name": "mean"},
        "yAxis": {"type": "value", "name": "std"},
        "series": [
            {
                "type": "scatter",
                "label": {"show": True, "formatter": "{b}", "position": "top"},
                "data": points,
            }
        ],
    }


def _binary_labels(data: dict[str, list[Any]]) -> list[int] | None:
    source = None
    if "label_action_type" in data:
        source = data["label_action_type"]
    elif "label_type" in data:
        source = data["label_type"]
    if source is None:
        return None
    labels: list[int] = []
    for value in source:
        scalar = _normalize_scalar(value)
        labels.append(1 if scalar == 2 else 0)
    return labels


def _label_distribution(data: dict[str, list[Any]]) -> Counter[Any]:
    if "label_type" in data:
        source = data["label_type"]
    elif "label_action_type" in data:
        source = data["label_action_type"]
    else:
        return Counter()
    counter: Counter[Any] = Counter()
    for value in source:
        scalar = _normalize_scalar(value)
        if scalar is not None:
            counter[scalar] += 1
    return counter


def _column_null_rows(feature_columns: list[str], data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        null_rate = _column_null_rate(data[column])
        rows.append({"name": column, "null_rate": round(null_rate, 6)})
    rows.sort(key=lambda item: item["null_rate"], reverse=True)
    return rows


def _sparse_cardinality_rows(sparse_columns: list[str], data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in sparse_columns:
        token_set: set[Any] = set()
        for value in data[column]:
            token_set.update(_sparse_tokens(value))
        rows.append({"name": column, "cardinality": len(token_set)})
    rows.sort(key=lambda item: item["cardinality"], reverse=True)
    return rows


def _sequence_length_stats(layout: SchemaLayout, data: dict[str, list[Any]], columns: set[str]) -> tuple[list[dict[str, Any]], dict[str, list[bool]]]:
    stats: list[dict[str, Any]] = []
    presence_by_domain: dict[str, list[bool]] = {}
    for domain in layout.sequence_domains:
        length_column = domain.length_column
        if length_column is None or length_column not in columns:
            continue
        lengths = [len(_normalize_list(value)) for value in data[length_column]]
        sorted_lengths = sorted(float(length) for length in lengths)
        if not sorted_lengths:
            continue
        presence = [length > 0 for length in lengths]
        presence_by_domain[domain.name] = presence
        stats.append(
            {
                "domain": domain.name,
                "min": float(sorted_lengths[0]),
                "q1": _quantile(sorted_lengths, 0.25),
                "median": _quantile(sorted_lengths, 0.5),
                "q3": _quantile(sorted_lengths, 0.75),
                "max": float(sorted_lengths[-1]),
                "mean": round(sum(sorted_lengths) / len(sorted_lengths), 6),
                "p95": _quantile(sorted_lengths, 0.95),
                "empty_rate": round(sum(1 for length in lengths if length == 0) / len(lengths), 6),
            }
        )
    return stats, presence_by_domain


def _user_activity(layout: SchemaLayout, data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    user_column = layout.primary_user_id_column
    if user_column is None or user_column not in data:
        return []
    activity_counter: Counter[Any] = Counter()
    for value in data[user_column]:
        token = _hashable_value(value)
        if token is not None:
            activity_counter[token] += 1
    if not activity_counter:
        return []
    bucket_counter: Counter[str] = Counter()
    for count in activity_counter.values():
        label = str(count) if count < 20 else "20+"
        bucket_counter[label] += 1
    numeric_labels = sorted(int(label) for label in bucket_counter if label != "20+")
    rows = [{"bucket": str(label), "user_count": bucket_counter[str(label)]} for label in numeric_labels]
    if "20+" in bucket_counter:
        rows.append({"bucket": "20+", "user_count": bucket_counter["20+"]})
    return rows


def _cross_domain_overlap(layout: SchemaLayout, data: dict[str, list[Any]], presence_by_domain: dict[str, list[bool]]) -> list[dict[str, Any]]:
    if not presence_by_domain:
        return []
    user_column = layout.primary_user_id_column
    user_domain_sets: defaultdict[Any, set[str]] = defaultdict(set)
    if user_column is not None and user_column in data:
        for index, value in enumerate(data[user_column]):
            token = _hashable_value(value)
            if token is None:
                token = index
            for domain, flags in presence_by_domain.items():
                if flags[index]:
                    user_domain_sets[token].add(domain)
    else:
        for index in range(len(next(iter(presence_by_domain.values())))):
            for domain, flags in presence_by_domain.items():
                if flags[index]:
                    user_domain_sets[index].add(domain)

    domains = sorted(presence_by_domain)
    users_by_domain: dict[str, set[Any]] = {domain: set() for domain in domains}
    for user_token, active_domains in user_domain_sets.items():
        for domain in active_domains:
            users_by_domain[domain].add(user_token)
    rows: list[dict[str, Any]] = []
    for left in domains:
        for right in domains:
            union = users_by_domain[left] | users_by_domain[right]
            overlap = 0.0 if not union else len(users_by_domain[left] & users_by_domain[right]) / float(len(union))
            rows.append({"left": left, "right": right, "overlap": round(overlap, 6)})
    return rows


def _dense_distribution_rows(dense_columns: list[str], data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in dense_columns:
        values: list[float] = []
        zero_count = 0
        for raw_value in data[column]:
            for item in _normalize_list(raw_value):
                number = float(item)
                values.append(number)
                if number == 0.0:
                    zero_count += 1
        if not values:
            rows.append({"name": column, "mean": 0.0, "std": 0.0, "zero_frac": 1.0})
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        rows.append(
            {
                "name": column,
                "mean": round(mean, 6),
                "std": round(math.sqrt(variance), 6),
                "zero_frac": round(zero_count / len(values), 6),
            }
        )
    return rows


def _cardinality_bins(cardinality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = {
        "1-10": 0,
        "11-100": 0,
        "101-1000": 0,
        "1001+": 0,
    }
    for row in cardinality_rows:
        cardinality = int(row["cardinality"])
        if cardinality <= 10:
            bins["1-10"] += 1
        elif cardinality <= 100:
            bins["11-100"] += 1
        elif cardinality <= 1000:
            bins["101-1000"] += 1
        else:
            bins["1001+"] += 1
    return [{"name": name, "count": count} for name, count in bins.items()]


def _seq_repeat_rate_rows(layout: SchemaLayout, data: dict[str, list[Any]], columns: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in layout.sequence_domains:
        repeat_column = domain.repeat_column
        if repeat_column is None or repeat_column not in columns:
            continue
        per_row_rates: list[float] = []
        for value in data[repeat_column]:
            tokens = [token for token in _normalize_list(value) if token not in (0, None)]
            if not tokens:
                continue
            unique_count = len(set(tokens))
            per_row_rates.append(1.0 - unique_count / float(len(tokens)))
        rows.append(
            {
                "domain": domain.name,
                "repeat_rate": round(sum(per_row_rates) / len(per_row_rates), 6) if per_row_rates else 0.0,
            }
        )
    return rows


def _co_missing_rows(null_rows: list[dict[str, Any]], data: dict[str, list[Any]]) -> tuple[list[str], list[list[Any]]]:
    feature_names = [row["name"] for row in null_rows[:12] if row["null_rate"] > 0.0]
    matrix: list[list[Any]] = []
    for y_index, left in enumerate(feature_names):
        left_missing = [_is_missing(value) for value in data[left]]
        for x_index, right in enumerate(feature_names):
            right_missing = [_is_missing(value) for value in data[right]]
            if not left_missing:
                rate = 0.0
            else:
                rate = sum(
                    1
                    for flag_left, flag_right in zip(left_missing, right_missing, strict=False)
                    if flag_left and flag_right
                ) / float(len(left_missing))
            matrix.append([x_index, y_index, round(rate, 6)])
    return feature_names, matrix


def _null_rate_by_label_rows(feature_columns: list[str], data: dict[str, list[Any]], labels: list[int]) -> list[dict[str, Any]]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    rows: list[dict[str, Any]] = []
    if positive_count == 0 or negative_count == 0:
        return rows
    for column in feature_columns:
        positive_missing = 0
        negative_missing = 0
        for value, label in zip(data[column], labels, strict=False):
            if not _is_missing(value):
                continue
            if label == 1:
                positive_missing += 1
            else:
                negative_missing += 1
        pos_rate = positive_missing / float(positive_count)
        neg_rate = negative_missing / float(negative_count)
        rows.append(
            {
                "name": column,
                "positive_missing_rate": round(pos_rate, 6),
                "negative_missing_rate": round(neg_rate, 6),
                "gap": round(abs(pos_rate - neg_rate), 6),
            }
        )
    rows.sort(key=lambda item: item["gap"], reverse=True)
    return rows


def _feature_auc_rows(sparse_columns: list[str], data: dict[str, list[Any]], labels: list[int]) -> list[dict[str, Any]]:
    if len(set(labels)) < 2:
        return []

    from sklearn.metrics import roc_auc_score

    global_rate = sum(labels) / float(len(labels))
    rows: list[dict[str, Any]] = []
    for column in sparse_columns:
        counts: defaultdict[Any, list[int]] = defaultdict(lambda: [0, 0])
        tokens = [_hashable_value(value) for value in data[column]]
        for token, label in zip(tokens, labels, strict=False):
            if token is None:
                continue
            counts[token][0] += label
            counts[token][1] += 1
        scores: list[float] = []
        for token in tokens:
            if token is None or token not in counts:
                scores.append(global_rate)
                continue
            positive, count = counts[token]
            scores.append((positive + global_rate) / (count + 1.0))
        if len(set(round(score, 8) for score in scores)) < 2:
            continue
        auc = roc_auc_score(labels, scores)
        rows.append({"name": column, "auc": round(float(auc), 6)})
    rows.sort(key=lambda item: item["auc"], reverse=True)
    return rows


def build_report(
    *,
    dataset_path: Path,
    schema_path: Path,
    role: DatasetRole,
    output_path: Path,
    chart_dir: Path | None,
    max_rows: int | None,
) -> dict[str, Any]:
    layout = SchemaLayout.from_path(schema_path)
    dataset = _load_dataset(
        dataset_path,
        columns=(*layout.feature_columns, "label_type", "label_action_type"),
        max_rows=max_rows,
    )
    row_count = dataset.table.num_rows
    subtitle = _base_subtitle(role, row_count, dataset.total_rows)
    data = {column: dataset.table.column(column).to_pylist() for column in dataset.table.column_names}
    feature_columns = [column for column in layout.feature_columns if column in data]
    sparse_columns = [column for column in layout.sparse_columns if column in data]
    dense_columns = [column for column in layout.user_dense_columns if column in data]
    labels = _binary_labels(data)
    label_columns = _detect_label_columns(set(data))
    label_dependent_enabled = role is DatasetRole.TEST and labels is not None and len(set(labels)) > 1

    group_map = layout.group_by_column
    scalar_columns = sorted(column for column in dataset.available_columns if column not in set(layout.feature_columns))
    layout_counts = {
        "scalar": len(scalar_columns),
        "user_int": len(layout.user_int_columns),
        "user_dense": len(layout.user_dense_columns),
        "item_int": len(layout.item_int_columns),
        "sequence": len(layout.sequence_columns),
    }
    domain_counts = {domain.name: len(domain.all_columns) for domain in layout.sequence_domains}

    null_rows = _column_null_rows(feature_columns, data)
    cardinality_rows = _sparse_cardinality_rows(sparse_columns, data)
    sequence_rows, presence_by_domain = _sequence_length_stats(layout, data, set(data))
    activity_rows = _user_activity(layout, data)
    overlap_rows = _cross_domain_overlap(layout, data, presence_by_domain)
    dense_rows = _dense_distribution_rows(dense_columns, data)
    cardinality_bin_rows = _cardinality_bins(cardinality_rows)
    repeat_rows = _seq_repeat_rate_rows(layout, data, set(data))
    co_missing_names, co_missing_matrix = _co_missing_rows(null_rows, data)

    coverage_x = [row["name"] for row in null_rows]
    coverage_y = sorted({group_map.get(name, "scalar") for name in coverage_x})
    y_index = {name: index for index, name in enumerate(coverage_y)}
    coverage_matrix = [
        [index, y_index[group_map.get(row["name"], "scalar")], round(1.0 - float(row["null_rate"]), 6)]
        for index, row in enumerate(null_rows)
    ]

    charts: dict[str, dict[str, Any]] = {
        "column_layout": _column_layout_chart(layout_counts, domain_counts, subtitle),
        "null_rates": _series_bar(
            [row["name"] for row in null_rows[:30]],
            [row["null_rate"] for row in null_rows[:30]],
            title="特征缺失率",
            subtitle=subtitle,
            series_name="null_rate",
            horizontal=True,
        ),
        "cardinality": _series_bar(
            [row["name"] for row in cardinality_rows[:20]],
            [row["cardinality"] for row in cardinality_rows[:20]],
            title="稀疏特征基数",
            subtitle=subtitle,
            series_name="cardinality",
            horizontal=True,
        ),
        "coverage_heatmap": _heatmap_chart(
            title="特征覆盖率热力图",
            subtitle=subtitle,
            x_labels=coverage_x,
            y_labels=coverage_y,
            data=coverage_matrix,
            value_name="coverage",
        ),
        "sequence_lengths": _boxplot_chart(
            [row["domain"] for row in sequence_rows],
            [[row["min"], row["q1"], row["median"], row["q3"], row["max"]] for row in sequence_rows],
            subtitle,
        ),
        "seq_length_summary": _sequence_summary_chart(
            [row["domain"] for row in sequence_rows],
            [row["mean"] for row in sequence_rows],
            [row["p95"] for row in sequence_rows],
            [row["empty_rate"] for row in sequence_rows],
            subtitle,
        ),
        "user_activity": _series_bar(
            [row["bucket"] for row in activity_rows],
            [row["user_count"] for row in activity_rows],
            title="用户活跃度分布",
            subtitle=subtitle,
            series_name="user_count",
        ),
        "cross_domain_overlap": _heatmap_chart(
            title="跨域用户重叠",
            subtitle=subtitle,
            x_labels=sorted(presence_by_domain),
            y_labels=sorted(presence_by_domain),
            data=[
                [
                    sorted(presence_by_domain).index(row["right"]),
                    sorted(presence_by_domain).index(row["left"]),
                    row["overlap"],
                ]
                for row in overlap_rows
            ],
            value_name="jaccard",
        ),
        "co_missing": _heatmap_chart(
            title="特征共缺失模式",
            subtitle=subtitle,
            x_labels=co_missing_names,
            y_labels=co_missing_names,
            data=co_missing_matrix,
            value_name="co_missing",
        ),
        "dense_distributions": _scatter_chart(
            title="稠密特征分布摘要",
            subtitle=subtitle,
            points=[
                {
                    "name": row["name"],
                    "value": [row["mean"], row["std"], max(12, int(16 + row["zero_frac"] * 28))],
                    "symbolSize": max(12, int(16 + row["zero_frac"] * 28)),
                }
                for row in dense_rows
            ],
        ),
        "cardinality_bins": _series_bar(
            [row["name"] for row in cardinality_bin_rows],
            [row["count"] for row in cardinality_bin_rows],
            title="特征基数区间分布",
            subtitle=subtitle,
            series_name="feature_count",
        ),
        "seq_repeat_rate": _series_bar(
            [row["domain"] for row in repeat_rows],
            [row["repeat_rate"] for row in repeat_rows],
            title="序列内物品重复率",
            subtitle=subtitle,
            series_name="repeat_rate",
        ),
    }

    if label_columns and role is DatasetRole.TEST:
        distribution = _label_distribution(data)
        charts["label_distribution"] = _label_distribution_chart(distribution, subtitle)
    if label_dependent_enabled and labels is not None:
        feature_auc_rows = _feature_auc_rows(sparse_columns, data, labels)
        null_by_label_rows = _null_rate_by_label_rows(feature_columns, data, labels)
        charts["feature_auc"] = _series_bar(
            [row["name"] for row in feature_auc_rows[:20]],
            [row["auc"] for row in feature_auc_rows[:20]],
            title="单特征 AUC 排名",
            subtitle=subtitle,
            series_name="auc",
            horizontal=True,
        )
        charts["null_rate_by_label"] = _grouped_bar_chart(
            title="正负样本缺失率对比",
            subtitle=subtitle,
            names=[row["name"] for row in null_by_label_rows[:20]],
            left_name="positive",
            left_values=[row["positive_missing_rate"] for row in null_by_label_rows[:20]],
            right_name="negative",
            right_values=[row["negative_missing_rate"] for row in null_by_label_rows[:20]],
        )
    else:
        feature_auc_rows = []
        null_by_label_rows = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_manifest: dict[str, str] = {}
    if chart_dir is not None:
        chart_dir.mkdir(parents=True, exist_ok=True)
        for chart_name, option in charts.items():
            chart_path = chart_dir / f"{chart_name}.echarts.json"
            _write_json(chart_path, option)
            chart_manifest[chart_name] = str(chart_path)

    skipped_charts = [name for name in TEST_ONLY_CHARTS if name not in charts]
    report = {
        "report": "dataset_eda",
        "dataset_path": str(dataset.dataset_path),
        "schema_path": str(schema_path),
        "dataset_role": role.value,
        "label_columns": list(label_columns),
        "label_dependent_analyses_enabled": label_dependent_enabled,
        "row_count": row_count,
        "total_rows": dataset.total_rows,
        "sampled": row_count != dataset.total_rows,
        "max_rows": max_rows,
        "chart_dir": str(chart_dir) if chart_dir is not None else None,
        "generated_charts": chart_manifest,
        "skipped_charts": skipped_charts,
        "stats": {
            "column_layout": {"counts": layout_counts, "domain_counts": domain_counts, "scalar_columns": scalar_columns},
            "null_rates": null_rows,
            "cardinality": cardinality_rows,
            "sequence_lengths": sequence_rows,
            "user_activity": activity_rows,
            "cross_domain_overlap": overlap_rows,
            "dense_distributions": dense_rows,
            "cardinality_bins": cardinality_bin_rows,
            "seq_repeat_rate": repeat_rows,
            "co_missing": {"columns": co_missing_names},
            "feature_auc": feature_auc_rows,
            "null_rate_by_label": null_by_label_rows,
            "label_distribution": dict(_label_distribution(data)),
        },
    }
    _write_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    files = _list_parquet_files(dataset_path)
    available_columns = set(_collect_available_columns(files))
    role, _label_columns = resolve_dataset_role(args.dataset_role, available_columns)
    resolved_schema_path = resolve_schema_path(dataset_path, Path(args.schema_path) if args.schema_path else None, Path.cwd())
    output_path, chart_dir = _resolve_output_paths(
        role=role,
        output_arg=args.output,
        json_path_arg=args.json_path,
        chart_dir_arg=args.chart_dir,
        no_charts=args.no_charts,
    )
    report = build_report(
        dataset_path=dataset_path,
        schema_path=resolved_schema_path,
        role=role,
        output_path=output_path,
        chart_dir=chart_dir,
        max_rows=args.max_rows,
    )
    print(output_path)
    if chart_dir is not None:
        print(chart_dir)
    if report["skipped_charts"]:
        print("skipped:", ", ".join(report["skipped_charts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
