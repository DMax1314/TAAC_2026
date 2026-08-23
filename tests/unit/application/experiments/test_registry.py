from __future__ import annotations

from pathlib import Path

import pytest

from taac2026.application.experiments.registry import load_experiment_package


@pytest.mark.parametrize("identifier", ["experiments/baseline", Path("experiments/baseline")])
def test_load_baseline_experiment_from_path(identifier: str | Path) -> None:
    experiment = load_experiment_package(identifier)

    assert experiment.name == "pcvr_hyformer"
    assert experiment.package_dir is not None
