"""Typed output contract for PCVR checkpoint learning traces."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from taac2026.domain.validation import TAACBoundaryModel


LearningCategory = Literal["early", "late", "unstable", "unlearned"]


class LearningTraceCheckpoint(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    step: int
    checkpoint_path: str
    auc: float
    logloss: float
    score_mean: float
    positive_reference_score: float
    negative_reference_score: float


class LearningTraceSample(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    sample_index: int
    user_id: str
    timestamp: int
    target: float
    scores: list[float]
    losses: list[float]
    learned_states: list[bool]
    first_sustained_step: int | None
    forgetting_events: int
    category: LearningCategory
    loss_improvement: float
    structural_features: dict[str, float]


class LearningTraceCategoryProfile(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    category: LearningCategory
    sample_count: int
    positive_count: int
    positive_rate: float
    loss_improvement_mean: float
    structural_feature_means: dict[str, float]


class LearningTraceReport(TAACBoundaryModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    run_dir: str
    dataset_path: str
    schema_path: str
    learning_rule: str
    steps: list[int]
    sample_count: int
    positive_count: int
    category_counts: dict[str, int]
    category_counts_by_target: dict[str, dict[str, int]]
    category_profiles: list[LearningTraceCategoryProfile]
    checkpoints: list[LearningTraceCheckpoint]
    samples: list[LearningTraceSample]
    report_path: str
    samples_path: str
    figure_path: str


__all__ = [
    "LearningCategory",
    "LearningTraceCategoryProfile",
    "LearningTraceCheckpoint",
    "LearningTraceReport",
    "LearningTraceSample",
]
