"""Trace fixed validation samples across PCVR training checkpoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np
import torch
import tyro

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from taac2026.application.evaluation.runtime import load_train_config
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.application.training.workflow import PCVRTrainContext, build_train_data, build_train_model
from taac2026.domain.learning_trace import (
    LearningCategory,
    LearningTraceCategoryProfile,
    LearningTraceCheckpoint,
    LearningTraceReport,
    LearningTraceSample,
)
from taac2026.domain.metrics import binary_auc, binary_logloss
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.checkpoints import (
    PRIMARY_CHECKPOINT_FILENAME,
    checkpoint_step,
    load_checkpoint_state_dict,
)
from taac2026.infrastructure.data.sample_dataset import resolve_default_pcvr_sample_paths
from taac2026.infrastructure.io.files import write_json
from taac2026.infrastructure.io.json import dump_bytes, dumps
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line
from taac2026.infrastructure.logging import configure_logging, logger
from taac2026.infrastructure.modeling.model_contract import resolve_checkpoint_schema_path
from taac2026.infrastructure.modeling.tensors import sigmoid_probabilities_numpy
from taac2026.infrastructure.runtime.execution import runtime_autocast_context
from taac2026.infrastructure.runtime.reporting import NoopTrainReporter


_CATEGORIES: tuple[LearningCategory, ...] = ("early", "late", "unstable", "unlearned")
_LEARNING_RULE = "opposite_class_median"
_EPSILON = 1.0e-7


@dataclass(frozen=True, slots=True)
class LearningTraceArgs:
    experiment: str
    run_dir: Path
    dataset_path: Path | None = None
    schema_path: Path | None = None
    output_dir: Path | None = None
    batch_size: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = False
    json: bool = False


@dataclass(frozen=True, slots=True)
class CheckpointPredictions:
    step: int
    checkpoint_path: Path
    labels: np.ndarray
    scores: np.ndarray
    user_ids: tuple[str, ...]
    timestamps: tuple[int, ...]
    structural_features: tuple[dict[str, float], ...] = ()


def discover_checkpoints(run_dir: Path) -> list[Path]:
    resolved_run_dir = run_dir.expanduser().resolve()
    checkpoints = sorted(
        resolved_run_dir.glob(f"global_step*/{PRIMARY_CHECKPOINT_FILENAME}"),
        key=checkpoint_step,
    )
    if len(checkpoints) < 2:
        raise ValueError(
            f"learning trace requires at least two step checkpoints under {resolved_run_dir}; "
            "train with --data.eval_every_n_steps smaller than --optimizer.max_steps"
        )
    steps = [checkpoint_step(path) for path in checkpoints]
    if any(step < 0 for step in steps):
        raise ValueError(f"invalid global_step checkpoint directory found under {resolved_run_dir}")
    if len(steps) != len(set(steps)):
        raise ValueError(f"duplicate checkpoint steps found under {resolved_run_dir}: {steps}")
    return checkpoints


def classify_learning_states(states: Sequence[bool], steps: Sequence[int]) -> tuple[LearningCategory, int | None, int]:
    if not states or len(states) != len(steps):
        raise ValueError("learning states and checkpoint steps must be non-empty and aligned")
    forgetting_events = sum(bool(previous) and not bool(current) for previous, current in pairwise(states))
    first_sustained_index = next(
        (index for index in range(len(states)) if all(states[index:])),
        None,
    )
    first_sustained_step = None if first_sustained_index is None else int(steps[first_sustained_index])
    if forgetting_events:
        return "unstable", first_sustained_step, forgetting_events
    if first_sustained_index is None:
        return "unlearned", None, 0
    early_boundary = max(0, (len(states) - 1) // 3)
    category: LearningCategory = "early" if first_sustained_index <= early_boundary else "late"
    return category, first_sustained_step, 0


def _binary_losses(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    probabilities = np.clip(scores.astype(np.float64), _EPSILON, 1.0 - _EPSILON)
    return -(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))


def _learned_states(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    if not positive_mask.any() or not negative_mask.any():
        raise ValueError("learning trace requires both positive and negative validation samples")
    positive_reference = float(np.median(scores[positive_mask]))
    negative_reference = float(np.median(scores[negative_mask]))
    learned = np.where(positive_mask, scores > negative_reference, scores < positive_reference)
    return learned.astype(bool), positive_reference, negative_reference


def analyze_checkpoint_predictions(
    predictions: Sequence[CheckpointPredictions],
    *,
    experiment_name: str,
    run_dir: Path,
    dataset_path: Path,
    schema_path: Path,
    report_path: Path,
    samples_path: Path,
    figure_path: Path,
) -> LearningTraceReport:
    if len(predictions) < 2:
        raise ValueError("learning trace requires at least two checkpoint predictions")
    ordered = sorted(predictions, key=lambda item: item.step)
    reference = ordered[0]
    sample_count = int(reference.labels.size)
    if sample_count == 0:
        raise ValueError("validation split produced no samples")
    for item in ordered:
        if item.scores.size != sample_count or len(item.user_ids) != sample_count or len(item.timestamps) != sample_count:
            raise ValueError(f"validation prediction fields are not aligned at checkpoint step {item.step}")
        if item.structural_features and len(item.structural_features) != sample_count:
            raise ValueError(f"validation structural features are not aligned at checkpoint step {item.step}")
    for item in ordered[1:]:
        if item.labels.shape != reference.labels.shape or not np.array_equal(item.labels, reference.labels):
            raise ValueError(f"validation labels changed at checkpoint step {item.step}")
        if item.user_ids != reference.user_ids or item.timestamps != reference.timestamps:
            raise ValueError(f"validation sample order changed at checkpoint step {item.step}")
        if item.structural_features != reference.structural_features:
            raise ValueError(f"validation structural features changed at checkpoint step {item.step}")

    steps = [item.step for item in ordered]
    learned_by_checkpoint: list[np.ndarray] = []
    losses_by_checkpoint: list[np.ndarray] = []
    checkpoint_summaries: list[LearningTraceCheckpoint] = []
    for item in ordered:
        learned, positive_reference, negative_reference = _learned_states(item.labels, item.scores)
        losses = _binary_losses(item.labels, item.scores)
        learned_by_checkpoint.append(learned)
        losses_by_checkpoint.append(losses)
        checkpoint_summaries.append(
            LearningTraceCheckpoint(
                step=item.step,
                checkpoint_path=str(item.checkpoint_path),
                auc=binary_auc(item.labels, item.scores),
                logloss=binary_logloss(item.labels, item.scores),
                score_mean=float(np.mean(item.scores)),
                positive_reference_score=positive_reference,
                negative_reference_score=negative_reference,
            )
        )

    samples: list[LearningTraceSample] = []
    category_counts: Counter[str] = Counter()
    target_counts: dict[str, Counter[str]] = {"negative": Counter(), "positive": Counter()}
    for sample_index in range(sample_count):
        scores = [float(item.scores[sample_index]) for item in ordered]
        losses = [float(values[sample_index]) for values in losses_by_checkpoint]
        states = [bool(values[sample_index]) for values in learned_by_checkpoint]
        category, first_sustained_step, forgetting_events = classify_learning_states(states, steps)
        target = float(reference.labels[sample_index])
        target_group = "positive" if target > 0.5 else "negative"
        category_counts[category] += 1
        target_counts[target_group][category] += 1
        samples.append(
            LearningTraceSample(
                sample_index=sample_index,
                user_id=reference.user_ids[sample_index],
                timestamp=reference.timestamps[sample_index],
                target=target,
                scores=scores,
                losses=losses,
                learned_states=states,
                first_sustained_step=first_sustained_step,
                forgetting_events=forgetting_events,
                category=category,
                loss_improvement=losses[0] - losses[-1],
                structural_features=dict(reference.structural_features[sample_index])
                if reference.structural_features
                else {},
            )
        )

    category_profiles = [_category_profile(category, samples) for category in _CATEGORIES]

    return LearningTraceReport(
        experiment_name=experiment_name,
        run_dir=str(run_dir),
        dataset_path=str(dataset_path),
        schema_path=str(schema_path),
        learning_rule=_LEARNING_RULE,
        steps=steps,
        sample_count=sample_count,
        positive_count=int((reference.labels > 0.5).sum()),
        category_counts={category: int(category_counts[category]) for category in _CATEGORIES},
        category_counts_by_target={
            target: {category: int(counts[category]) for category in _CATEGORIES}
            for target, counts in target_counts.items()
        },
        category_profiles=category_profiles,
        checkpoints=checkpoint_summaries,
        samples=samples,
        report_path=str(report_path),
        samples_path=str(samples_path),
        figure_path=str(figure_path),
    )


def _category_profile(
    category: LearningCategory,
    samples: Sequence[LearningTraceSample],
) -> LearningTraceCategoryProfile:
    selected = [sample for sample in samples if sample.category == category]
    positive_count = sum(sample.target > 0.5 for sample in selected)
    feature_names = sorted({name for sample in selected for name in sample.structural_features})
    feature_means = {
        name: float(np.mean([sample.structural_features[name] for sample in selected if name in sample.structural_features]))
        for name in feature_names
    }
    return LearningTraceCategoryProfile(
        category=category,
        sample_count=len(selected),
        positive_count=positive_count,
        positive_rate=float(positive_count / len(selected)) if selected else 0.0,
        loss_improvement_mean=float(np.mean([sample.loss_improvement for sample in selected])) if selected else 0.0,
        structural_feature_means=feature_means,
    )


def _batch_structural_features(batch: Any) -> list[dict[str, float]]:
    row_count = len(batch.user_id)
    features = [dict() for _ in range(row_count)]
    total_sequence_lengths = np.zeros(row_count, dtype=np.float64)
    for domain, sequence in sorted(batch.inputs.sequences.items()):
        lengths = sequence.lengths.detach().cpu().numpy().astype(np.float64)
        total_sequence_lengths += lengths
        for row_index, value in enumerate(lengths.tolist()):
            features[row_index][f"sequence_length/{domain}"] = float(value)
    for row_index, value in enumerate(total_sequence_lengths.tolist()):
        features[row_index]["sequence_length/total"] = float(value)

    missing_masks = {
        "missing_ratio/user_int": batch.inputs.user.int_missing_mask,
        "missing_ratio/user_dense": batch.inputs.user.dense_missing_mask,
        "missing_ratio/item_int": batch.inputs.item.int_missing_mask,
        "missing_ratio/item_dense": batch.inputs.item.dense_missing_mask,
    }
    for name, mask in missing_masks.items():
        if mask.ndim < 2 or mask.shape[1] == 0:
            continue
        ratios = mask.detach().float().mean(dim=1).cpu().numpy()
        for row_index, value in enumerate(ratios.tolist()):
            features[row_index][name] = float(value)
    return features


def _collect_predictions(
    *,
    checkpoint: Path,
    model: torch.nn.Module,
    valid_loader: Any,
    device: torch.device,
    runtime: RuntimeExecutionConfig,
) -> CheckpointPredictions:
    state_dict = load_checkpoint_state_dict(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    user_ids: list[str] = []
    timestamps: list[int] = []
    structural_features: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch in valid_loader:
            model_input = batch.inputs.to(device)
            with runtime_autocast_context(runtime, device):
                logits, _embeddings = model.predict(model_input)
            probabilities = sigmoid_probabilities_numpy(logits.squeeze(-1))
            labels.extend(float(value) for value in batch.label.detach().cpu().numpy().tolist())
            scores.extend(float(value) for value in probabilities.tolist())
            user_ids.extend(str(value) for value in batch.user_id)
            timestamps.extend(int(value) for value in batch.inputs.request_timestamp.detach().cpu().tolist())
            structural_features.extend(_batch_structural_features(batch))
    return CheckpointPredictions(
        step=checkpoint_step(checkpoint),
        checkpoint_path=checkpoint,
        labels=np.asarray(labels, dtype=np.float64),
        scores=np.asarray(scores, dtype=np.float64),
        user_ids=tuple(user_ids),
        timestamps=tuple(timestamps),
        structural_features=tuple(structural_features),
    )


def _write_samples(path: Path, samples: Sequence[LearningTraceSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for sample in samples:
            handle.write(dump_bytes(sample.model_dump(mode="json")))
            handle.write(b"\n")


def _plot_report(report: LearningTraceReport, output_path: Path) -> None:
    steps = np.asarray(report.steps, dtype=np.int64)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    figure.suptitle(f"Learning Trace: {report.experiment_name}", fontsize=14, fontweight="bold")

    axes[0, 0].plot(steps, [item.auc for item in report.checkpoints], marker="o", color="#2563eb")
    axes[0, 0].set_title("Validation AUC")
    axes[0, 0].set_xlabel("checkpoint step")

    axes[0, 1].plot(steps, [item.logloss for item in report.checkpoints], marker="o", color="#dc2626")
    axes[0, 1].set_title("Validation LogLoss")
    axes[0, 1].set_xlabel("checkpoint step")

    colors = {"early": "#16a34a", "late": "#2563eb", "unstable": "#f59e0b", "unlearned": "#dc2626"}
    for category in _CATEGORIES:
        category_samples = [sample for sample in report.samples if sample.category == category]
        if not category_samples:
            continue
        median_losses = np.median(np.asarray([sample.losses for sample in category_samples]), axis=0)
        axes[1, 0].plot(
            steps,
            median_losses,
            marker="o",
            label=f"{category} (n={len(category_samples)})",
            color=colors[category],
        )
    axes[1, 0].set_title("Median Per-Sample BCE")
    axes[1, 0].set_xlabel("checkpoint step")
    axes[1, 0].legend(fontsize=8)

    labels = ["negative", "positive"]
    bottoms = np.zeros(len(labels), dtype=np.float64)
    for category in _CATEGORIES:
        values = np.asarray([report.category_counts_by_target[label][category] for label in labels], dtype=np.float64)
        axes[1, 1].bar(labels, values, bottom=bottoms, label=category, color=colors[category])
        bottoms += values
    axes[1, 1].set_title("Learning Categories by Target")
    axes[1, 1].set_ylabel("samples")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def run_learning_trace(args: LearningTraceArgs) -> LearningTraceReport:
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or (run_dir / "learning_trace")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "learning_trace.log")
    checkpoints = discover_checkpoints(run_dir)
    experiment = load_experiment_package(args.experiment)
    if getattr(experiment, "kind", None) != "pcvr":
        raise TypeError(f"learning trace requires a PCVR experiment, got {getattr(experiment, 'kind', None)!r}")
    model_type = getattr(experiment, "model_type", None)
    config_type = getattr(experiment, "config_type", None)
    if model_type is None or config_type is None:
        raise TypeError("PCVR experiment does not expose model_type and config_type")

    dataset_path, schema_override = resolve_default_pcvr_sample_paths(args.dataset_path, args.schema_path)
    schema_path = resolve_checkpoint_schema_path(checkpoints[0].parent, schema_override)
    config = load_train_config(config_type, checkpoints[0].parent)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    batch_size = args.batch_size if args.batch_size is not None else config.data.batch_size
    config = config.model_copy(update={"data": config.data.model_copy(update={"batch_size": batch_size, "num_workers": 0})})
    device = torch.device(args.device)
    runtime = RuntimeExecutionConfig(
        amp=args.amp,
        amp_dtype=config.runtime.amp_dtype,
        compile=False,
        deterministic=config.runtime.deterministic,
    )
    context = PCVRTrainContext(
        model_class_name=model_type.__name__,
        model_type=model_type,
        config=config,
        data_dir=dataset_path,
        ckpt_dir=run_dir,
        schema_path=schema_path,
        device=str(device),
        reporter=NoopTrainReporter(),
    )
    data_bundle = build_train_data(context)
    model = build_train_model(context, data_bundle)
    prediction_sets: list[CheckpointPredictions] = []
    for checkpoint in checkpoints:
        logger.info("Tracing checkpoint step {}: {}", checkpoint_step(checkpoint), checkpoint)
        prediction_sets.append(
            _collect_predictions(
                checkpoint=checkpoint,
                model=model,
                valid_loader=data_bundle.valid_loader,
                device=device,
                runtime=runtime,
            )
        )

    report_path = output_dir / "learning_trace.json"
    samples_path = output_dir / "learning_trace_samples.jsonl"
    figure_path = output_dir / "learning_trace.svg"
    report = analyze_checkpoint_predictions(
        prediction_sets,
        experiment_name=getattr(experiment, "name", args.experiment),
        run_dir=run_dir,
        dataset_path=dataset_path,
        schema_path=schema_path,
        report_path=report_path,
        samples_path=samples_path,
        figure_path=figure_path,
    )
    _write_samples(samples_path, report.samples)
    write_json(report_path, report.model_dump(mode="json"))
    _plot_report(report, figure_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> LearningTraceArgs:
    return tyro.cli(LearningTraceArgs, description=__doc__, args=argv)


def _format_summary(report: LearningTraceReport) -> None:
    checkpoint_fields = [
        (f"step {item.step}", f"AUC={item.auc:.5f}, LogLoss={item.logloss:.5f}")
        for item in report.checkpoints
    ]
    category_fields = [(category, str(report.category_counts[category])) for category in _CATEGORIES]
    print_rich_summary(
        "PCVR learning trace complete",
        [
            ("Experiment", report.experiment_name),
            ("Samples", str(report.sample_count)),
            ("Checkpoints", str(len(report.steps))),
            ("Learning rule", report.learning_rule),
            ("Report", report.report_path),
            ("Figure", report.figure_path),
        ],
        sections=(("Checkpoint metrics", checkpoint_fields), ("Learning categories", category_fields)),
        border_style="magenta",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_learning_trace(args)
    if args.json:
        write_stdout_line(dumps(report.model_dump(mode="json")))
    else:
        _format_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
