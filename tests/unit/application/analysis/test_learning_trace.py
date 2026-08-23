from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from taac2026.application.analysis.learning_trace import (
    CheckpointPredictions,
    analyze_checkpoint_predictions,
    classify_learning_states,
    discover_checkpoints,
)


def test_discover_checkpoints_sorts_by_global_step(tmp_path: Path) -> None:
    for directory_name in ("global_step20.AUC=0.7", "global_step5.AUC=0.6", "global_step10.AUC=0.65"):
        checkpoint_dir = tmp_path / directory_name
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")

    checkpoints = discover_checkpoints(tmp_path)

    assert [path.parent.name for path in checkpoints] == [
        "global_step5.AUC=0.6",
        "global_step10.AUC=0.65",
        "global_step20.AUC=0.7",
    ]


def test_discover_checkpoints_requires_multiple_steps(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "global_step10"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="at least two"):
        discover_checkpoints(tmp_path)


def test_discover_checkpoints_rejects_invalid_global_step_directory(tmp_path: Path) -> None:
    for directory_name in ("global_step_bad", "global_step10"):
        checkpoint_dir = tmp_path / directory_name
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="invalid global_step"):
        discover_checkpoints(tmp_path)


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        ([True, True, True, True], ("early", 5, 0)),
        ([False, False, True, True], ("late", 15, 0)),
        ([True, False, True, True], ("unstable", 15, 1)),
        ([False, False, False, False], ("unlearned", None, 0)),
    ],
)
def test_classify_learning_states(states: list[bool], expected: tuple[str, int | None, int]) -> None:
    assert classify_learning_states(states, [5, 10, 15, 20]) == expected


def test_analyze_checkpoint_predictions_builds_aligned_sample_traces(tmp_path: Path) -> None:
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    metadata = {
        "labels": labels,
        "user_ids": ("u0", "u1", "u2", "u3"),
        "timestamps": (10, 11, 12, 13),
        "structural_features": (
            {"sequence_length/total": 1.0},
            {"sequence_length/total": 2.0},
            {"sequence_length/total": 3.0},
            {"sequence_length/total": 4.0},
        ),
    }
    predictions = [
        CheckpointPredictions(
            step=5,
            checkpoint_path=tmp_path / "global_step5" / "model.safetensors",
            scores=np.asarray([0.1, 0.4, 0.3, 0.8]),
            **metadata,
        ),
        CheckpointPredictions(
            step=10,
            checkpoint_path=tmp_path / "global_step10" / "model.safetensors",
            scores=np.asarray([0.1, 0.2, 0.6, 0.9]),
            **metadata,
        ),
    ]

    report = analyze_checkpoint_predictions(
        predictions,
        experiment_name="pcvr_test",
        run_dir=tmp_path,
        dataset_path=tmp_path / "data.parquet",
        schema_path=tmp_path / "schema.json",
        report_path=tmp_path / "learning_trace.json",
        samples_path=tmp_path / "learning_trace_samples.jsonl",
        figure_path=tmp_path / "learning_trace.svg",
    )

    assert report.steps == [5, 10]
    assert report.sample_count == 4
    assert report.positive_count == 2
    assert sum(report.category_counts.values()) == 4
    assert len(report.samples) == 4
    assert report.checkpoints[1].auc == pytest.approx(1.0)
    assert report.checkpoints[1].logloss < report.checkpoints[0].logloss
    assert sum(profile.sample_count for profile in report.category_profiles) == 4
    assert report.samples[0].structural_features == {"sequence_length/total": 1.0}
    profiles = {profile.category: profile for profile in report.category_profiles}
    assert profiles["early"].sample_count == 4
    assert profiles["early"].structural_feature_means["sequence_length/total"] == pytest.approx(2.5)
    assert profiles["late"].structural_feature_means == {}
