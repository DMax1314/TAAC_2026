"""Generate PCVR smoke-test diagnostic figures from local run outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
from typing import Any, Literal

import matplotlib
import numpy as np
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.io.json import dumps, loads, read_path
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line


FIGURE_NAMES = {
    "runtime_resources": "pcvr_runtime_resources.svg",
    "prediction_distribution": "pcvr_prediction_distribution.svg",
    "prediction_correlation": "pcvr_prediction_correlation.svg",
    "sample_disagreement": "pcvr_sample_disagreement.svg",
    "stability": "pcvr_stability.svg",
}
PLOT_BG = "#101418"
AX_BG = "#151a20"
GRID = "#2b333c"
TEXT = "#d8dee9"
SUBTEXT = "#9aa7b8"
COLORS = ["#4c8bf5", "#00b894", "#f6c85f", "#ef6f6c", "#9b8cff", "#4dd0e1", "#ff9f43"]
MODEL_COLORS = ["#6baed6", "#f6bd60", "#7fc97f", "#f28e8c", "#b39ddb", "#80cbc4", "#ffcc80"]
DiagnosticGroupBy = Literal["experiment", "label", "label-prefix"]


@dataclass(slots=True)
class PCVRDiagnosticPlotArgs:
    run: list[str]
    output_dir: Path = Path("figures/pcvr_diagnostics")
    bins: int = 24
    top_disagreement: int = 20
    allow_partial: bool = False
    json: bool = False
    group_by: DiagnosticGroupBy = "experiment"


@dataclass(slots=True)
class PredictionRecord:
    key: str
    score: float
    target: float | None
    user_id: str | None
    sample_index: int | None


@dataclass(slots=True)
class DiagnosticRun:
    label: str
    group: str
    run_dir: Path
    evaluation_path: Path | None
    predictions_path: Path | None
    evaluation: dict[str, Any]
    training_summary: dict[str, Any]
    training_telemetry: dict[str, Any]
    evaluation_telemetry: dict[str, Any]
    inference_telemetry: dict[str, Any]
    predictions: list[PredictionRecord]


class DiagnosticInputError(ValueError):
    """Raised when requested diagnostic figures cannot be built from inputs."""


def _parse_run_spec(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if label.strip() and path.strip():
            return label.strip(), Path(path).expanduser()
    return None, Path(value).expanduser()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_path(path)
    return payload if isinstance(payload, dict) else {}


def _resolve_output_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [base_dir / path, Path.cwd() / path]
    candidates.extend(parent / path for parent in base_dir.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return (base_dir / path).resolve()


def _prediction_key(record: dict[str, Any], row_index: int) -> str:
    sample_index = record.get("sample_index")
    if sample_index is not None:
        return f"sample:{sample_index}"
    user_id = record.get("user_id")
    if user_id is not None:
        return f"user:{user_id}"
    return f"row:{row_index}"


def _load_predictions(path: Path | None) -> list[PredictionRecord]:
    if path is None or not path.exists():
        return []
    records: list[PredictionRecord] = []
    for row_index, line in enumerate(path.read_bytes().splitlines()):
        if not line.strip():
            continue
        payload = loads(line)
        if not isinstance(payload, dict) or "score" not in payload:
            continue
        sample_index = payload.get("sample_index")
        try:
            sample_index_value = int(sample_index) if sample_index is not None else None
        except (TypeError, ValueError):
            sample_index_value = None
        target = payload.get("target")
        records.append(
            PredictionRecord(
                key=_prediction_key(payload, row_index),
                score=float(payload["score"]),
                target=float(target) if target is not None else None,
                user_id=str(payload["user_id"]) if payload.get("user_id") is not None else None,
                sample_index=sample_index_value,
            )
        )
    return records


def _group_label(label: str, experiment_name: str | None, strategy: str) -> str:
    if strategy == "label":
        return label
    if strategy == "experiment" and experiment_name:
        return experiment_name
    if strategy == "label-prefix":
        lowered = label.lower()
        for marker in ("_seed", "-seed", ".seed", "_run", "-run", ".run"):
            index = lowered.find(marker)
            if index > 0:
                return label[:index]
    return experiment_name or label


def _load_run(spec: str, *, group_by: str) -> DiagnosticRun:
    explicit_label, raw_path = _parse_run_spec(spec)
    resolved_path = raw_path.resolve()
    run_dir = resolved_path.parent if resolved_path.is_file() else resolved_path
    evaluation_path = resolved_path if resolved_path.is_file() else run_dir / "evaluation.json"
    evaluation = _read_json_if_exists(evaluation_path)
    if not evaluation:
        evaluation_path = None
    training_summary = _read_json_if_exists(run_dir / "training_summary.json")
    training_telemetry = _read_json_if_exists(run_dir / "training_telemetry.json")
    evaluation_telemetry = _read_json_if_exists(run_dir / "evaluation_telemetry.json")
    inference_telemetry = _read_json_if_exists(run_dir / "inference_telemetry.json")
    if not training_telemetry:
        training_telemetry = _dict_value(training_summary, "telemetry")
    if not evaluation_telemetry:
        evaluation_telemetry = _dict_value(evaluation, "telemetry")
    predictions_path = _resolve_output_path(
        run_dir,
        _string_value(evaluation, "validation_predictions_path"),
    ) or (run_dir / "validation_predictions.jsonl")
    predictions = _load_predictions(predictions_path)
    experiment_name = _string_value(evaluation, "experiment_name") or _string_value(training_summary, "experiment_name")
    label = explicit_label or experiment_name or run_dir.name
    return DiagnosticRun(
        label=label,
        group=_group_label(label, experiment_name, group_by),
        run_dir=run_dir,
        evaluation_path=evaluation_path,
        predictions_path=predictions_path if predictions_path.exists() else None,
        evaluation=evaluation,
        training_summary=training_summary,
        training_telemetry=training_telemetry,
        evaluation_telemetry=evaluation_telemetry,
        inference_telemetry=inference_telemetry,
        predictions=predictions,
    )


def _run_missing_required_inputs(run: DiagnosticRun) -> list[str]:
    missing = []
    if run.evaluation_path is None:
        missing.append(str(run.run_dir / "evaluation.json"))
    if run.predictions_path is None:
        missing.append(str(run.run_dir / "validation_predictions.jsonl"))
    return missing


def _run_warnings(run: DiagnosticRun) -> list[str]:
    warnings = []
    if not run.training_telemetry:
        warnings.append("training telemetry missing")
    if not run.evaluation_telemetry:
        warnings.append("evaluation telemetry missing")
    if not _comparison_metrics(run):
        warnings.append("validation metrics missing")
    return warnings


def _infer_experiment_path(run: DiagnosticRun) -> str:
    package_dir = _string_value(_dict_value(run.training_telemetry, "metadata"), "package_dir")
    if package_dir:
        path = Path(package_dir).expanduser()
        try:
            return str(path.resolve().relative_to(Path.cwd()))
        except ValueError:
            return str(path)
    name = run.group or run.label
    for marker in ("_seed", "-seed", ".seed", "_run", "-run", ".run"):
        index = name.lower().find(marker)
        if index > 0:
            name = name[:index]
            break
    cleaned = name.removeprefix("pcvr_").replace("_", "-")
    return f"experiments/{cleaned}"


def _format_missing_inputs(runs: list[DiagnosticRun]) -> str:
    lines = [
        "PCVR diagnostic inputs are incomplete.",
        "",
        "Missing required files:",
    ]
    for run in runs:
        missing = _run_missing_required_inputs(run)
        if not missing:
            continue
        lines.append(f"  - {run.label} ({run.run_dir}):")
        for path in missing:
            lines.append(f"      {path}")
    lines.extend(
        [
            "",
            "Run validation first, for example:",
        ]
    )
    for run in runs:
        if _run_missing_required_inputs(run):
            lines.append(f"  bash run.sh val --experiment {_infer_experiment_path(run)} --run-dir {run.run_dir}")
    lines.extend(
        [
            "",
            "Use --allow-partial only when you intentionally want placeholder figures.",
        ]
    )
    return "\n".join(lines)


def _validate_required_inputs(runs: list[DiagnosticRun]) -> None:
    if any(_run_missing_required_inputs(run) for run in runs):
        raise DiagnosticInputError(_format_missing_inputs(runs))


def _string_value(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _dict_value(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _telemetry(run: DiagnosticRun, phase: str) -> dict[str, Any]:
    if phase == "training":
        return run.training_telemetry
    if phase == "evaluation":
        return run.evaluation_telemetry
    if phase == "inference":
        return run.inference_telemetry
    raise ValueError(f"unsupported phase: {phase}")


def _metric(run: DiagnosticRun, name: str) -> float | None:
    return _nested_float(_comparison_metrics(run), name)


def _comparison_metrics(run: DiagnosticRun) -> dict[str, Any]:
    validation_metrics = _dict_value(run.training_summary, "validation_metrics")
    if validation_metrics:
        metrics = dict(validation_metrics)
        score_diagnostics = _dict_value(run.training_summary, "validation_score_diagnostics")
        if score_diagnostics:
            metrics["score_diagnostics"] = score_diagnostics
        return metrics
    return _dict_value(run.evaluation, "metrics")


def _comparison_metric_source(run: DiagnosticRun) -> str | None:
    if _dict_value(run.training_summary, "validation_metrics"):
        return "training_validation"
    if _dict_value(run.evaluation, "metrics"):
        return "evaluation"
    return None


def _configure_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _new_figure(width: float, height: float) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "figure.facecolor": PLOT_BG,
            "savefig.facecolor": PLOT_BG,
        }
    )
    return plt.figure(figsize=(width, height), dpi=160)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _message_figure(output_path: Path, title: str, message: str) -> None:
    fig = _new_figure(8.0, 4.2)
    ax = fig.add_subplot(111)
    ax.set_facecolor(AX_BG)
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, color=SUBTEXT)
    ax.set_axis_off()
    _save(fig, output_path)


def _grouped_runs(runs: list[DiagnosticRun]) -> list[tuple[str, list[DiagnosticRun]]]:
    groups: dict[str, list[DiagnosticRun]] = {}
    for run in runs:
        groups.setdefault(run.group or run.label, []).append(run)
    return list(groups.items())


def _group_means(grouped_runs: list[tuple[str, list[DiagnosticRun]]], getter) -> np.ndarray:
    means = []
    for _label, group_runs in grouped_runs:
        values = [value for run in group_runs if (value := getter(run)) is not None and np.isfinite(value)]
        means.append(float(np.mean(values)) if values else float("nan"))
    return np.asarray(means, dtype=float)


def _run_peak(run: DiagnosticRun, key: str) -> float | None:
    values = [
        value
        for phase in ("training", "evaluation", "inference")
        if (value := _nested_float(_telemetry(run, phase), key)) is not None and np.isfinite(value)
    ]
    return max(values) if values else None


def plot_runtime_resources(runs: list[DiagnosticRun], output_path: Path) -> None:
    grouped_runs = _grouped_runs(runs)
    if not grouped_runs:
        _message_figure(output_path, "Runtime / Resource", "no run outputs were provided")
        return
    labels = [label for label, _group_runs in grouped_runs]
    train_elapsed = _group_means(grouped_runs, lambda run: _nested_float(run.training_telemetry, "elapsed_sec"))
    eval_elapsed = _group_means(grouped_runs, lambda run: _nested_float(run.evaluation_telemetry, "elapsed_sec"))
    infer_elapsed = _group_means(grouped_runs, lambda run: _nested_float(run.inference_telemetry, "elapsed_sec"))
    eval_throughput = _group_means(grouped_runs, lambda run: _nested_float(run.evaluation_telemetry, "rows_per_sec"))
    infer_throughput = _group_means(grouped_runs, lambda run: _nested_float(run.inference_telemetry, "rows_per_sec"))
    cpu_peak = _group_means(grouped_runs, lambda run: _run_peak(run, "cpu_peak_rss_mb")) / 1024.0
    cuda_peak = _group_means(grouped_runs, lambda run: _run_peak(run, "cuda_peak_allocated_mb")) / 1024.0
    parameter_counts = _group_means(
        grouped_runs,
        lambda run: (_nested_float(run.training_telemetry, "model_parameters") or float("nan")) / 1_000_000.0,
    )

    metric_specs = [
        ("train s", train_elapsed, True, lambda value: f"{value:.1f}"),
        ("eval s", eval_elapsed, True, lambda value: f"{value:.1f}"),
        ("eval r/s", eval_throughput, False, lambda value: f"{value:.0f}"),
        ("CPU GB", cpu_peak, True, lambda value: f"{value:.1f}"),
        ("CUDA GB", cuda_peak, True, lambda value: f"{value:.1f}"),
        ("params M", parameter_counts, True, lambda value: f"{value:.0f}"),
    ]
    has_inference = np.isfinite(infer_elapsed).all() and np.isfinite(infer_throughput).all()
    if has_inference:
        metric_specs.insert(2, ("infer s", infer_elapsed, True, lambda value: f"{value:.1f}"))
        metric_specs.insert(4, ("infer r/s", infer_throughput, False, lambda value: f"{value:.0f}"))

    scorecard_labels = [label for label, _values, _lower, _formatter in metric_specs]
    scorecard = np.column_stack([values for _label, values, _lower, _formatter in metric_specs])
    lower_is_better = np.asarray([lower for _label, _values, lower, _formatter in metric_specs], dtype=bool)
    formatted = np.column_stack(
        [
            [formatter(value) if np.isfinite(value) else "-" for value in values]
            for _label, values, _lower, formatter in metric_specs
        ]
    )
    heatmap = np.column_stack(
        [_normalized_metric(scorecard[:, index], lower_is_better=bool(lower_is_better[index])) for index in range(scorecard.shape[1])]
    )

    fig = _new_figure(12.8, 7.8)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], width_ratios=[1.0, 1.0])
    ax_scorecard = fig.add_subplot(grid[0, :])
    ax_train_tradeoff = fig.add_subplot(grid[1, 0])
    ax_infer_tradeoff = fig.add_subplot(grid[1, 1])

    _plot_runtime_scorecard(ax_scorecard, labels, scorecard_labels, formatted, heatmap)
    _plot_tradeoff_scatter(
        ax_train_tradeoff,
        labels,
        x=train_elapsed,
        y=cuda_peak,
        sizes=parameter_counts,
        title="Training Cost Tradeoff (bubble area ~= parameters)",
        xlabel="train runtime (s)",
        ylabel="peak CUDA allocated (GB)",
    )
    efficiency_phase = "Inference" if has_inference else "Evaluation"
    _plot_tradeoff_scatter(
        ax_infer_tradeoff,
        labels,
        x=infer_throughput if has_inference else eval_throughput,
        y=infer_elapsed if has_inference else eval_elapsed,
        sizes=cuda_peak,
        title=f"{efficiency_phase} Efficiency Tradeoff (bubble area ~= peak CUDA)",
        xlabel=f"{efficiency_phase.lower()} throughput (rows/s)",
        ylabel=f"{efficiency_phase.lower()} runtime (s)",
    )
    fig.suptitle("Model-Level Runtime / Resource Comparison", color=TEXT, fontsize=15, fontweight="bold")
    _save(fig, output_path)


def _normalized_metric(values: np.ndarray, *, lower_is_better: bool) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    low = float(finite.min())
    high = float(finite.max())
    if high <= low:
        return np.full_like(values, 0.5, dtype=float)
    scaled = (values - low) / (high - low)
    return 1.0 - scaled if lower_is_better else scaled


def _plot_runtime_scorecard(
    ax: plt.Axes,
    labels: list[str],
    metric_labels: list[str],
    formatted: np.ndarray,
    heatmap: np.ndarray,
) -> None:
    ax.set_facecolor(AX_BG)
    image = ax.imshow(heatmap, aspect="auto", vmin=0.0, vmax=1.0, cmap="cividis")
    del image
    ax.set_title("Runtime / Resource Scorecard", color=TEXT, pad=14)
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, color=SUBTEXT, fontsize=9)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, color=SUBTEXT, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    for row in range(formatted.shape[0]):
        for col in range(formatted.shape[1]):
            text_color = "#101418" if np.isfinite(heatmap[row, col]) and heatmap[row, col] > 0.68 else "#f8f9fb"
            ax.text(col, row, formatted[row, col], ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
    ax.set_xlabel("brighter is better within each column", color=SUBTEXT, labelpad=10)


def _plot_tradeoff_scatter(
    ax: plt.Axes,
    labels: list[str],
    *,
    x: np.ndarray,
    y: np.ndarray,
    sizes: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    _configure_axes(ax)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=SUBTEXT, transform=ax.transAxes)
        ax.set_title(title)
        return
    labels = [label for label, keep in zip(labels, finite, strict=True) if keep]
    x = x[finite]
    y = y[finite]
    sizes = sizes[finite]
    finite_sizes = sizes[np.isfinite(sizes) & (sizes > 0.0)]
    size_scale = float(finite_sizes.max()) if finite_sizes.size else 1.0
    marker_sizes = 90.0 + 430.0 * np.sqrt(np.nan_to_num(sizes, nan=0.0) / size_scale)
    for row, label in enumerate(labels):
        ax.scatter(
            [x[row]],
            [y[row]],
            s=[marker_sizes[row]],
            color=MODEL_COLORS[row % len(MODEL_COLORS)],
            edgecolors="#f8f9fb",
            linewidths=0.55,
            alpha=0.88,
            label=label,
            zorder=3,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    legend = ax.legend(frameon=False, labelcolor=SUBTEXT, fontsize=7, ncols=2, loc="best")
    for handle in legend.legend_handles:
        handle.set_sizes([55.0])


def _prediction_maps(runs: list[DiagnosticRun]) -> list[tuple[DiagnosticRun, dict[str, PredictionRecord]]]:
    return [(run, {record.key: record for record in run.predictions}) for run in runs if run.predictions]


def _grouped_prediction_maps(runs: list[DiagnosticRun]) -> list[tuple[str, dict[str, PredictionRecord]]]:
    grouped_maps = []
    for label, group_runs in _grouped_runs(runs):
        run_maps = [records for _run, records in _prediction_maps(group_runs)]
        if not run_maps:
            continue
        common_keys = sorted(set.intersection(*(set(records) for records in run_maps)))
        aggregated = {}
        for key in common_keys:
            source = run_maps[0][key]
            target = next((records[key].target for records in run_maps if records[key].target is not None), None)
            aggregated[key] = PredictionRecord(
                key=key,
                score=float(np.mean([records[key].score for records in run_maps])),
                target=target,
                user_id=source.user_id,
                sample_index=source.sample_index,
            )
        grouped_maps.append((label, aggregated))
    return grouped_maps


def plot_prediction_distribution(runs: list[DiagnosticRun], output_path: Path, *, bins: int) -> None:
    maps = _grouped_prediction_maps(runs)
    if not maps:
        _message_figure(output_path, "Prediction Distribution", "no validation_predictions.jsonl files were found")
        return
    all_scores = np.asarray([record.score for _label, records in maps for record in records.values()], dtype=float)
    x_max = min(1.0, max(0.25, float(all_scores.max()) * 1.05))
    bin_edges = np.linspace(0.0, x_max, max(2, bins) + 1)
    ncols = 2 if len(maps) > 1 else 1
    nrows = int(np.ceil(len(maps) / ncols))
    fig = _new_figure(11.5 if ncols == 2 else 6.0, 2.7 * nrows + 0.8)
    grid = GridSpec(nrows, 4 if ncols == 2 else 1, figure=fig)
    fig.suptitle("Class-Conditional Prediction Profiles", color=TEXT, fontsize=15, fontweight="bold")
    for index, (label, records) in enumerate(maps, start=1):
        row = (index - 1) // ncols
        if ncols == 2 and len(maps) % 2 == 1 and index == len(maps):
            ax = fig.add_subplot(grid[row, 1:3])
        elif ncols == 2:
            column = ((index - 1) % 2) * 2
            ax = fig.add_subplot(grid[row, column : column + 2])
        else:
            ax = fig.add_subplot(grid[row, 0])
        _configure_axes(ax)
        scores = np.asarray([record.score for record in records.values()], dtype=float)
        targets = np.asarray(
            [record.target if record.target is not None else np.nan for record in records.values()],
            dtype=float,
        )
        negative_scores = scores[targets < 0.5]
        positive_scores = scores[targets >= 0.5]
        has_labels = negative_scores.size > 0 and positive_scores.size > 0
        if has_labels:
            ax.hist(
                negative_scores,
                bins=bin_edges,
                weights=np.full(negative_scores.size, 1.0 / negative_scores.size),
                color=COLORS[3],
                alpha=0.58,
                label="target=0",
            )
            ax.hist(
                positive_scores,
                bins=bin_edges,
                weights=np.full(positive_scores.size, 1.0 / positive_scores.size),
                color=COLORS[1],
                alpha=0.62,
                label="target=1",
            )
            margin = float(positive_scores.mean() - negative_scores.mean())
            ax.set_title(f"{label} | mean margin={margin:+.3f}")
            if index == 1:
                ax.legend(frameon=False, labelcolor=SUBTEXT, fontsize=8)
        else:
            ax.hist(scores, bins=bin_edges, weights=np.full(scores.size, 1.0 / scores.size), color=COLORS[0], alpha=0.8)
            ax.set_title(label)
        ax.set_xlim(0.0, x_max)
        ax.set_xlabel("seed-mean predicted probability")
        ax.set_ylabel("within-class share")
    _save(fig, output_path)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _pairwise_spearman(left: dict[str, PredictionRecord], right: dict[str, PredictionRecord]) -> tuple[float, int]:
    keys = sorted(set(left).intersection(right))
    if len(keys) < 2:
        return float("nan"), len(keys)
    left_scores = _rankdata(np.asarray([left[key].score for key in keys], dtype=float))
    right_scores = _rankdata(np.asarray([right[key].score for key in keys], dtype=float))
    if left_scores.std() == 0.0 or right_scores.std() == 0.0:
        return float("nan"), len(keys)
    return float(np.corrcoef(left_scores, right_scores)[0, 1]), len(keys)


def _within_group_seed_agreement(runs: list[DiagnosticRun]) -> list[tuple[str, float, int]]:
    agreements = []
    for label, group_runs in _grouped_runs(runs):
        maps = [records for _run, records in _prediction_maps(group_runs)]
        correlations = [
            correlation
            for left_index, left in enumerate(maps)
            for right in maps[left_index + 1 :]
            if np.isfinite(correlation := _pairwise_spearman(left, right)[0])
        ]
        agreements.append((label, float(np.mean(correlations)) if correlations else float("nan"), len(maps)))
    return agreements


def plot_prediction_correlation(runs: list[DiagnosticRun], output_path: Path) -> None:
    maps = _grouped_prediction_maps(runs)
    if len(maps) < 2:
        _message_figure(output_path, "Prediction Correlation", "need at least two model groups with validation predictions")
        return
    labels = [label for label, _records in maps]
    size = len(maps)
    matrix = np.full((size, size), np.nan, dtype=float)
    overlaps = np.zeros((size, size), dtype=int)
    for i, (_left_label, left_records) in enumerate(maps):
        for j, (_right_label, right_records) in enumerate(maps):
            if i == j:
                matrix[i, j] = 1.0
                overlaps[i, j] = len(left_records)
            elif i < j:
                corr, overlap = _pairwise_spearman(left_records, right_records)
                matrix[i, j] = matrix[j, i] = corr
                overlaps[i, j] = overlaps[j, i] = overlap
    fig = _new_figure(12.5, 6.6)
    grid = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 0.65])
    ax = fig.add_subplot(grid[0, 0])
    ax.set_facecolor(AX_BG)
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Between-Model Ranking Agreement")
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels(labels, rotation=35, ha="right", color=SUBTEXT)
    ax.set_yticklabels(labels, color=SUBTEXT)
    for i in range(size):
        for j in range(size):
            value = matrix[i, j]
            text = "nan" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color="#f8f9fb", fontsize=9, fontweight="bold")
    common_overlap = int(overlaps[np.triu_indices(size, 1)].min())
    ax.set_xlabel(f"Spearman rho on seed-mean predictions; common samples >= {common_overlap}", color=SUBTEXT)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=SUBTEXT)
    ax_agreement = fig.add_subplot(grid[0, 1])
    _configure_axes(ax_agreement)
    agreements = _within_group_seed_agreement(runs)
    agreement_labels = [label for label, _value, _count in agreements]
    agreement_values = np.asarray([value for _label, value, _count in agreements], dtype=float)
    y = np.arange(len(agreements))
    ax_agreement.barh(y, agreement_values, color=MODEL_COLORS[: len(agreements)], alpha=0.86)
    ax_agreement.set_yticks(y)
    ax_agreement.set_yticklabels(agreement_labels, color=SUBTEXT)
    ax_agreement.invert_yaxis()
    finite_agreements = agreement_values[np.isfinite(agreement_values)]
    agreement_min = min(0.0, float(finite_agreements.min()) * 1.05) if finite_agreements.size else 0.0
    ax_agreement.set_xlim(agreement_min, 1.0)
    ax_agreement.set_title("Within-Model Seed Agreement")
    ax_agreement.set_xlabel("mean pairwise Spearman rho")
    for row, (_label, value, count) in enumerate(agreements):
        text = f"{value:.3f} ({count} seeds)" if np.isfinite(value) else "single seed"
        text_x = value - 0.02 if np.isfinite(value) else agreement_min + 0.02
        ax_agreement.text(text_x, row, text, va="center", ha="right", color=TEXT, fontsize=8)
    fig.suptitle("Seed-Aware Prediction Correlation", color=TEXT, fontsize=15, fontweight="bold")
    _save(fig, output_path)


def plot_sample_disagreement(runs: list[DiagnosticRun], output_path: Path, *, top_n: int) -> None:
    maps = _grouped_prediction_maps(runs)
    if len(maps) < 2:
        _message_figure(output_path, "Sample Disagreement", "need at least two model groups with validation predictions")
        return
    common_keys = sorted(set.intersection(*(set(records) for _label, records in maps)))
    if not common_keys:
        _message_figure(output_path, "Sample Disagreement", "runs do not share prediction sample keys")
        return
    scores = np.asarray([[records[key].score for _run, records in maps] for key in common_keys], dtype=float)
    mean_scores = scores.mean(axis=1)
    std_scores = scores.std(axis=1)
    range_scores = scores.max(axis=1) - scores.min(axis=1)
    targets = np.asarray(
        [next((records[key].target for _label, records in maps if records[key].target is not None), np.nan) for key in common_keys],
        dtype=float,
    )
    top_count = min(max(1, top_n), 20, len(common_keys))
    order = np.argsort(std_scores)[::-1][:top_count]
    model_labels = [label for label, _records in maps]

    fig = _new_figure(12.8, max(7.8, 0.3 * top_count + 3.8))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[0.9, 3.2], width_ratios=[1.0, 1.65])
    ax_hist = fig.add_subplot(grid[0, :])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_heatmap = fig.add_subplot(grid[1, 1])

    _configure_axes(ax_hist)
    ax_hist.grid(True, axis="x", color=GRID, linewidth=0.7, alpha=0.75)
    bins = np.linspace(0.0, max(float(std_scores.max()) * 1.05, 1e-6), 32)
    ax_hist.hist(std_scores, bins=bins, color=COLORS[0], alpha=0.84)
    p50, p90, p99 = np.quantile(std_scores, [0.5, 0.9, 0.99])
    for value, label, color in ((p50, "p50", COLORS[2]), (p90, "p90", COLORS[1]), (p99, "p99", COLORS[3])):
        ax_hist.axvline(value, color=color, linewidth=1.3, alpha=0.9)
        ax_hist.text(value, 0.95, label, color=color, fontsize=8, ha="center", va="top", transform=ax_hist.get_xaxis_transform())
    ax_hist.set_title("Model Disagreement Across Validation Samples")
    ax_hist.set_xlabel("std across model-level seed-mean predictions")
    ax_hist.set_ylabel("samples")

    _configure_axes(ax_scatter)
    colors = np.where(np.nan_to_num(targets, nan=0.0) >= 0.5, COLORS[1], COLORS[3])
    ax_scatter.scatter(mean_scores, std_scores, c=colors, s=16, alpha=0.56, edgecolors="none")
    ax_scatter.scatter(
        mean_scores[order],
        std_scores[order],
        facecolors="none",
        edgecolors="#f8f9fb",
        linewidths=0.8,
        s=42,
        label=f"top {top_count}",
    )
    ax_scatter.scatter([], [], color=COLORS[3], s=24, label="target=0")
    ax_scatter.scatter([], [], color=COLORS[1], s=24, label="target=1")
    ax_scatter.set_title("All Samples: Mean vs Disagreement")
    ax_scatter.set_xlabel("mean predicted probability")
    ax_scatter.set_ylabel("std across models")
    ax_scatter.legend(frameon=False, labelcolor=SUBTEXT, fontsize=8, loc="upper left")

    top_scores = scores[order]
    heatmap = ax_heatmap.imshow(top_scores, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax_heatmap.set_facecolor(AX_BG)
    ax_heatmap.set_title(f"Top {top_count} Samples: Prediction By Model")
    ax_heatmap.set_xticks(np.arange(len(model_labels)))
    ax_heatmap.set_xticklabels(model_labels, rotation=25, ha="right", color=SUBTEXT)
    row_labels = [
        f"{_short_sample_label(common_keys[index])} | y={_target_label(targets[index])} | sd={std_scores[index]:.3f}"
        for index in order
    ]
    ax_heatmap.set_yticks(np.arange(top_count))
    ax_heatmap.set_yticklabels(row_labels, color=SUBTEXT, fontsize=8)
    ax_heatmap.tick_params(colors=SUBTEXT)
    for spine in ax_heatmap.spines.values():
        spine.set_color(GRID)
    for row in range(top_count):
        for col in range(len(model_labels)):
            value = top_scores[row, col]
            text_color = "#101418" if value > 0.58 else "#f8f9fb"
            ax_heatmap.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    colorbar = fig.colorbar(heatmap, ax=ax_heatmap, fraction=0.035, pad=0.025)
    colorbar.set_label("predicted probability", color=TEXT)
    colorbar.ax.tick_params(colors=SUBTEXT)
    ax_heatmap.set_xlabel(
        f"Cells are model means across seeds; max model range={range_scores[order[0]]:.3f}",
        color=SUBTEXT,
    )
    _save(fig, output_path)


def _short_sample_label(key: str) -> str:
    value = key.split(":", 1)[-1]
    return value if len(value) <= 18 else value[:15] + "..."


def _target_label(value: float) -> str:
    if not np.isfinite(value):
        return "?"
    return "1" if value >= 0.5 else "0"


def _metric_groups(runs: list[DiagnosticRun], getter) -> list[tuple[str, list[float]]]:
    groups: dict[str, list[float]] = {}
    for run in runs:
        value = getter(run)
        if value is None or not np.isfinite(value):
            continue
        groups.setdefault(run.group or run.label, []).append(float(value))
    return list(groups.items())


def _prediction_seed_drift_groups(runs: list[DiagnosticRun]) -> tuple[list[tuple[str, list[float]]], dict[str, int]]:
    groups = []
    seed_counts = {}
    for label, group_runs in _grouped_runs(runs):
        maps = [records for _run, records in _prediction_maps(group_runs)]
        seed_counts[label] = len(maps)
        if len(maps) < 2:
            continue
        common_keys = sorted(set.intersection(*(set(records) for records in maps)))
        if not common_keys:
            continue
        scores = np.asarray([[records[key].score for records in maps] for key in common_keys], dtype=float)
        mean_sample_sd = float(np.mean(np.std(scores, axis=1, ddof=1)))
        groups.append((label, [mean_sample_sd]))
    return groups, seed_counts


def _plot_smoke_metric_panel(
    ax: plt.Axes,
    title: str,
    groups: list[tuple[str, list[float]]],
    *,
    formatter,
    higher_is_better: bool | None,
    display_counts: dict[str, int] | None = None,
) -> None:
    _configure_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.7, alpha=0.75)
    ax.grid(False, axis="y")
    ax.set_title(title)
    if not groups:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=SUBTEXT, transform=ax.transAxes)
        return
    reverse = higher_is_better is not False
    ordered = sorted(groups, key=lambda item: float(np.mean(item[1])), reverse=reverse)
    labels = [label for label, _values in ordered]
    metric_values = np.asarray([float(np.mean(values)) for _label, values in ordered], dtype=float)
    metric_mins = np.asarray([float(np.min(values)) for _label, values in ordered], dtype=float)
    metric_maxes = np.asarray([float(np.max(values)) for _label, values in ordered], dtype=float)
    metric_stds = np.asarray(
        [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for _label, values in ordered],
        dtype=float,
    )
    counts = np.asarray([len(values) for _label, values in ordered], dtype=int)
    y = np.arange(len(ordered))
    colors = [COLORS[0]] * len(ordered)
    if higher_is_better is not None and len(colors) > 1:
        colors[0] = COLORS[1]
        colors[-1] = COLORS[3]
    ax.barh(y, metric_values, color=colors, alpha=0.86, height=0.62)
    if int(counts.max()) > 1:
        ax.errorbar(
            metric_values,
            y,
            xerr=np.vstack([metric_values - metric_mins, metric_maxes - metric_values]),
            fmt="none",
            ecolor="#f8f9fb",
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=SUBTEXT, fontsize=8)
    ax.invert_yaxis()
    max_value = float(metric_maxes.max())
    x_max = 1.0 if title.startswith("AUC") else max(max_value * 1.18, 1e-6)
    ax.set_xlim(0.0, x_max)
    label_offset = x_max * 0.012
    for row, value in enumerate(metric_values):
        label_x = min(value + label_offset, x_max * 0.985)
        ha = "left" if label_x > value else "right"
        display_count = display_counts.get(labels[row], counts[row]) if display_counts else counts[row]
        if counts[row] > 1:
            label = f"{formatter(value)} +/- {formatter(metric_stds[row])} ({display_count} seeds)"
        elif display_count > 1:
            label = f"{formatter(value)} ({display_count} seeds)"
        else:
            label = formatter(value)
        ax.text(label_x, row, label, ha=ha, va="center", color=TEXT, fontsize=8, fontweight="bold")
    if higher_is_better is True:
        ax.set_xlabel("higher is better")
    elif higher_is_better is False:
        ax.set_xlabel("lower is better")
    else:
        ax.set_xlabel("spread / variation")


def plot_stability(runs: list[DiagnosticRun], output_path: Path) -> None:
    if not runs:
        _message_figure(output_path, "Stability", "no run outputs were provided")
        return
    drift_groups, drift_seed_counts = _prediction_seed_drift_groups(runs)
    panels = [
        ("AUC", _metric_groups(runs, lambda run: _metric(run, "auc")), lambda value: f"{value:.4f}", True, None),
        (
            "LogLoss",
            _metric_groups(runs, lambda run: _metric(run, "logloss")),
            lambda value: f"{value:.4f}",
            False,
            None,
        ),
        ("Prediction Seed Drift", drift_groups, lambda value: f"{value:.4f}", False, drift_seed_counts),
        (
            "Eval Runtime",
            _metric_groups(runs, lambda run: _nested_float(run.evaluation_telemetry, "elapsed_sec")),
            lambda value: f"{value:.2f}s",
            False,
            None,
        ),
    ]
    repeated_groups = any(len(values) > 1 for _title, groups, _formatter, _higher, _counts in panels for _label, values in groups)
    fig = _new_figure(12.5, 8.2)
    title = "Three-Seed Smoke Comparison" if repeated_groups else "Single-Run Smoke Metrics"
    fig.suptitle(title, color=TEXT, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "bars = mean; whiskers = min-max; +/- = sample SD",
        ha="center",
        color=SUBTEXT,
        fontsize=9,
    )
    for index, (title, groups, formatter, higher_is_better, display_counts) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, index)
        _plot_smoke_metric_panel(
            ax,
            title,
            groups,
            formatter=formatter,
            higher_is_better=higher_is_better,
            display_counts=display_counts,
        )
    _save(fig, output_path)


def _run_summary(run: DiagnosticRun) -> dict[str, Any]:
    return {
        "label": run.label,
        "group": run.group,
        "run_dir": str(run.run_dir),
        "evaluation_path": str(run.evaluation_path) if run.evaluation_path else None,
        "predictions_path": str(run.predictions_path) if run.predictions_path else None,
        "prediction_count": len(run.predictions),
        "missing_required_inputs": _run_missing_required_inputs(run),
        "warnings": _run_warnings(run),
        "metrics": _comparison_metrics(run),
        "metric_source": _comparison_metric_source(run),
        "telemetry": {
            "training": run.training_telemetry,
            "evaluation": run.evaluation_telemetry,
            "inference": run.inference_telemetry,
        },
    }


def run_diagnostics(args: PCVRDiagnosticPlotArgs) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    runs = [_load_run(spec, group_by=args.group_by) for spec in args.run]
    if not args.allow_partial:
        _validate_required_inputs(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = {name: output_dir / filename for name, filename in FIGURE_NAMES.items()}
    plot_runtime_resources(runs, figure_paths["runtime_resources"])
    plot_prediction_distribution(runs, figure_paths["prediction_distribution"], bins=args.bins)
    plot_prediction_correlation(runs, figure_paths["prediction_correlation"])
    plot_sample_disagreement(runs, figure_paths["sample_disagreement"], top_n=args.top_disagreement)
    plot_stability(runs, figure_paths["stability"])
    summary = {
        "output_dir": str(output_dir),
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "runs": [_run_summary(run) for run in runs],
    }
    summary_path = output_dir / "pcvr_diagnostics_summary.json"
    summary["summary_path"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def _normalize_repeated_run_args(argv: Sequence[str] | None) -> Sequence[str]:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    runs: list[str] = []
    normalized: list[str] = []
    index = 0
    while index < len(raw_argv):
        arg = raw_argv[index]
        if arg == "--run" and index + 1 < len(raw_argv):
            runs.append(raw_argv[index + 1])
            index += 2
            continue
        if arg.startswith("--run="):
            runs.append(arg.split("=", 1)[1])
            index += 1
            continue
        normalized.append(arg)
        index += 1
    if runs:
        normalized.extend(("--run", *runs))
    return tuple(normalized)


def parse_args(argv: Sequence[str] | None = None) -> PCVRDiagnosticPlotArgs:
    return tyro.cli(
        PCVRDiagnosticPlotArgs,
        description=__doc__,
        args=_normalize_repeated_run_args(argv),
    )


def _format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{seconds:.2f}s"


def _format_rate(value: Any) -> str:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{rate:.1f}/s"


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return "-"


def format_summary(summary: dict[str, Any]) -> str:
    lines = [f"PCVR diagnostics written to: {summary['output_dir']}", ""]
    lines.append("Figures:")
    for name, path in summary["figures"].items():
        lines.append(f"  - {name}: {path}")
    lines.append("")
    lines.append("Runs:")
    for run in summary["runs"]:
        metrics = _dict_value(run, "metrics")
        telemetry = _dict_value(run, "telemetry")
        train_telemetry = _dict_value(telemetry, "training")
        eval_telemetry = _dict_value(telemetry, "evaluation")
        missing = run.get("missing_required_inputs") or []
        status = "ok" if not missing else "partial"
        lines.append(
            "  - "
            f"{run['label']} [{status}]: "
            f"predictions={run['prediction_count']}, "
            f"auc={_format_metric(metrics.get('auc'))}, "
            f"logloss={_format_metric(metrics.get('logloss'))}, "
            f"train={_format_seconds(train_telemetry.get('elapsed_sec'))}, "
            f"eval={_format_seconds(eval_telemetry.get('elapsed_sec'))}, "
            f"eval_rows={_format_rate(eval_telemetry.get('rows_per_sec'))}"
        )
        warnings = list(run.get("warnings") or [])
        if missing:
            warnings.append(f"missing {len(missing)} required file(s)")
        for warning in warnings:
            lines.append(f"      warning: {warning}")
    lines.append("")
    lines.append(f"Summary JSON: {summary['summary_path']}")
    return "\n".join(lines)


def _format_diagnostic_summary(summary: dict[str, Any]) -> None:
    figures = summary.get("figures") or {}
    fields = [("Output dir", str(summary.get("output_dir", "<unknown>")))]
    for name, path in figures.items():
        fields.append((name, str(path)))
    sections = []
    for run in summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        label = run.get("label", "<unknown>")
        metrics = _dict_value(run, "metrics")
        telemetry = _dict_value(run, "telemetry")
        train_telemetry = _dict_value(telemetry, "training")
        eval_telemetry = _dict_value(telemetry, "evaluation")
        missing = run.get("missing_required_inputs") or []
        status = "ok" if not missing else "partial"
        run_fields = [("Status", status)]
        auc = metrics.get("auc")
        logloss = metrics.get("logloss")
        if auc is not None:
            run_fields.append(("AUC", f"{auc:.5f}"))
        if logloss is not None:
            run_fields.append(("LogLoss", f"{logloss:.5f}"))
        train_sec = train_telemetry.get("elapsed_sec")
        eval_sec = eval_telemetry.get("elapsed_sec")
        if train_sec is not None:
            run_fields.append(("Train", f"{train_sec:.2f}s"))
        if eval_sec is not None:
            run_fields.append(("Eval", f"{eval_sec:.2f}s"))
        for warning in run.get("warnings") or []:
            run_fields.append(("Warning", str(warning)))
        sections.append((label, run_fields))
    print_rich_summary(
        "PCVR diagnostics generated",
        fields,
        sections=sections,
        subtitle=f"Summary JSON: {summary.get('summary_path', '<unknown>')}",
        border_style="magenta",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_diagnostics(args)
    except DiagnosticInputError as error:
        sys.stderr.write(f"{error}\n")
        return 2
    if args.json:
        write_stdout_line(dumps(summary, indent=2))
    else:
        _format_diagnostic_summary(summary)
    return 0


__all__ = [
    "DiagnosticInputError",
    "DiagnosticRun",
    "PredictionRecord",
    "format_summary",
    "parse_args",
    "run_diagnostics",
]
