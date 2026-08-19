"""Experiment plugin contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from taac2026.domain.requests import EvalRequest, InferRequest, TrainRequest


TrainFn = Callable[[TrainRequest], Mapping[str, Any] | None]
EvalFn = Callable[[EvalRequest], Mapping[str, Any]]
InferFn = Callable[[InferRequest], Mapping[str, Any]]


@runtime_checkable
class Experiment(Protocol):
    name: str
    package_dir: Path | None
    kind: str
    requires_dataset: bool
    train_defaults: Any | None

    def train(self, request: TrainRequest) -> Mapping[str, Any] | None: ...

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]: ...

    def infer(self, request: InferRequest) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class FunctionExperiment:
    name: str
    package_dir: Path | None = None
    kind: str = "maintenance"
    requires_dataset: bool = True
    train_fn: TrainFn | None = None
    evaluate_fn: EvalFn | None = None
    infer_fn: InferFn | None = None
    train_defaults: Any | None = None

    def train(self, request: TrainRequest) -> Mapping[str, Any] | None:
        if self.train_fn is None:
            raise NotImplementedError(f"experiment {self.name!r} does not implement training")
        return self.train_fn(request)

    def evaluate(self, request: EvalRequest) -> Mapping[str, Any]:
        if self.evaluate_fn is None:
            raise NotImplementedError(f"experiment {self.name!r} does not implement evaluation")
        return self.evaluate_fn(request)

    def infer(self, request: InferRequest) -> Mapping[str, Any]:
        if self.infer_fn is None:
            raise NotImplementedError(f"experiment {self.name!r} does not implement inference")
        return self.infer_fn(request)


__all__ = ["Experiment", "FunctionExperiment"]
