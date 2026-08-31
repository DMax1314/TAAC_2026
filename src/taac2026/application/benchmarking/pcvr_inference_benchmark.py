"""Benchmark end-to-end PCVR inference throughput and latency with a trained checkpoint."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

from taac2026.application.evaluation.runtime import load_runtime_schema, load_train_config
from taac2026.application.evaluation.workflow import (
    PCVRPredictionContext,
    PCVRPredictionDataBundle,
    PCVRPredictionRunner,
    build_prediction_data,
    build_prediction_model,
    predict_batch_probabilities,
    prepare_prediction_runner,
)
from taac2026.application.experiments.registry import load_experiment_package
from taac2026.domain.runtime_config import RuntimeExecutionConfig
from taac2026.infrastructure.checkpoints import resolve_checkpoint_path
from taac2026.infrastructure.data.sample_dataset import resolve_default_pcvr_sample_paths
from taac2026.infrastructure.io.json import dumps
from taac2026.infrastructure.io.rich_output import print_rich_summary
from taac2026.infrastructure.io.streams import write_stdout_line


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class PCVRInferenceBenchmarkArgs:
    experiment: str
    checkpoint: Path
    dataset_path: Path | None = None
    schema_path: Path | None = None
    batch_size: int = 256
    num_workers: int = 0
    device: str = field(default_factory=_default_device)
    amp: bool = False
    amp_dtype: str = "bfloat16"
    compile: bool = False
    warmup_batches: int = 3
    max_batches: int = 0
    repeats: int = 1
    json: bool = False


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _measured_inference_pass(
    context: PCVRPredictionContext,
    data_bundle: PCVRPredictionDataBundle,
    runner: PCVRPredictionRunner,
    *,
    warmup_batches: int,
    max_batches: int,
) -> dict[str, Any]:
    device = context.runtime_device
    predictions: dict[str, float] = {}
    batch_latencies_ms: list[float] = []
    batch_rows: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in data_bundle.loader:
            _sync_device(device)
            batch_started = time.perf_counter()
            batch_user_ids, batch_probabilities = predict_batch_probabilities(context, runner, batch)
            predictions.update(zip(map(str, batch_user_ids), batch_probabilities.tolist(), strict=False))
            _sync_device(device)
            batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)
            batch_rows.append(len(batch_probabilities))
            if max_batches > 0 and len(batch_rows) >= warmup_batches + max_batches:
                break
    elapsed_sec = time.perf_counter() - started

    rows = sum(batch_rows)
    steady_latencies_ms = batch_latencies_ms[warmup_batches:]
    steady_rows = sum(batch_rows[warmup_batches:])
    steady_sec = sum(steady_latencies_ms) / 1000.0
    return {
        "rows": rows,
        "batches": len(batch_rows),
        "elapsed_sec": elapsed_sec,
        "rows_per_sec": rows / elapsed_sec if elapsed_sec > 0 else 0.0,
        "batch_latency_ms": _latency_stats(batch_latencies_ms),
        "warmup_rows": sum(batch_rows[:warmup_batches]),
        "steady_rows": steady_rows,
        "steady_rows_per_sec": steady_rows / steady_sec if steady_sec > 0 else 0.0,
        "steady_batch_latency_ms": _latency_stats(steady_latencies_ms),
    }


def run_benchmark(args: PCVRInferenceBenchmarkArgs) -> dict[str, Any]:
    experiment = load_experiment_package(args.experiment)
    if getattr(experiment, "kind", None) != "pcvr":
        raise ValueError(f"experiment {args.experiment!r} is not a PCVR experiment")
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    dataset_path, schema_override = resolve_default_pcvr_sample_paths(args.dataset_path, args.schema_path)
    config = load_train_config(experiment.config_type, checkpoint.parent)
    schema_path, _schema = load_runtime_schema(
        schema_path=schema_override,
        checkpoint_dir=checkpoint.parent,
        mode="inference",
    )
    runtime_execution = RuntimeExecutionConfig(
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        compile=args.compile,
    )
    context = PCVRPredictionContext(
        model_type=experiment.model_type,
        dataset_path=dataset_path,
        schema_path=schema_path,
        checkpoint_path=checkpoint,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        is_training_data=False,
        dataset_role="inference",
        config=config,
        runtime_execution=runtime_execution,
    )

    data_started = time.perf_counter()
    data_bundle = build_prediction_data(context)
    data_build_sec = time.perf_counter() - data_started
    model_started = time.perf_counter()
    model = build_prediction_model(context, data_bundle)
    runner = prepare_prediction_runner(context, data_bundle, model)
    model_prepare_sec = time.perf_counter() - model_started

    repeats = max(1, int(args.repeats))
    passes = [
        {
            "repeat": repeat,
            **_measured_inference_pass(
                context,
                data_bundle,
                runner,
                warmup_batches=max(0, int(args.warmup_batches)),
                max_batches=max(0, int(args.max_batches)),
            ),
        }
        for repeat in range(repeats)
    ]

    return {
        "status": "ok",
        "experiment": experiment.name,
        "checkpoint": str(checkpoint),
        "dataset_path": str(dataset_path),
        "schema_path": str(schema_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": runtime_execution.amp,
        "amp_dtype": runtime_execution.normalized_amp_dtype(),
        "compile": runtime_execution.compile,
        "dataset_rows": int(getattr(data_bundle.dataset, "num_rows", 0)),
        "data_build_sec": data_build_sec,
        "model_prepare_sec": model_prepare_sec,
        "warmup_batches": args.warmup_batches,
        "max_batches": args.max_batches,
        "repeats": repeats,
        "passes": passes,
        "last_pass": passes[-1],
    }


def parse_args(argv: Sequence[str] | None = None) -> PCVRInferenceBenchmarkArgs:
    return tyro.cli(PCVRInferenceBenchmarkArgs, description=__doc__, args=argv)


def _format_inference_summary(summary: dict[str, Any]) -> None:
    last_pass = summary.get("last_pass") or {}
    fields = [
        ("Experiment", str(summary.get("experiment", "<unknown>"))),
        ("Checkpoint", str(summary.get("checkpoint", "<unknown>"))),
        ("Dataset rows", f"{summary.get('dataset_rows', 0):,}"),
        ("Device", str(summary.get("device", "<unknown>"))),
        ("Batch size", str(summary.get("batch_size", 0))),
        ("Num workers", str(summary.get("num_workers", 0))),
        ("AMP", f"{summary.get('amp')} ({summary.get('amp_dtype')})"),
        ("Compile", str(summary.get("compile"))),
    ]
    sections = [
        ("Setup", [
            ("Data build", f"{summary.get('data_build_sec', 0.0):.3f}s"),
            ("Model prepare", f"{summary.get('model_prepare_sec', 0.0):.3f}s"),
        ]),
        ("Last pass", [
            ("Rows", f"{last_pass.get('rows', 0):,}"),
            ("Batches", str(last_pass.get("batches", 0))),
            ("Elapsed", f"{last_pass.get('elapsed_sec', 0.0):.3f}s"),
            ("Rows/sec", f"{last_pass.get('rows_per_sec', 0.0):.1f}"),
            ("Steady rows/sec", f"{last_pass.get('steady_rows_per_sec', 0.0):.1f}"),
        ]),
    ]
    print_rich_summary("PCVR inference benchmark", fields, sections=sections, border_style="blue")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args)
    if args.json:
        write_stdout_line(dumps(summary, indent=2))
    else:
        _format_inference_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
