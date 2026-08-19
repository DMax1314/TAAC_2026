from __future__ import annotations

import shutil
from pathlib import Path

from tests.support.experiment_discovery import discover_experiment_paths
from tests.support.experiment_matrix import build_pcvr_experiment_cases
from tests.support.paths import fixture_path


def _copy_minimal_pcvr_experiment(package_dir: Path) -> None:
    shutil.copytree(fixture_path("experiments", "minimal_pcvr"), package_dir)


def test_discover_experiment_paths_filters_to_valid_packages(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments"
    experiment_root.mkdir(parents=True)

    valid = experiment_root / "valid_exp"
    valid.mkdir()
    for name in ("__init__.py", "model.py"):
        (valid / name).touch()

    hidden = experiment_root / "__pycache__"
    hidden.mkdir()
    (hidden / "__init__.py").touch()

    assert discover_experiment_paths(experiment_root) == ["experiments/valid_exp"]


def test_build_pcvr_experiment_cases_discovers_minimal_new_package(tmp_path: Path) -> None:
    experiment_root = tmp_path / "experiments"
    _copy_minimal_pcvr_experiment(experiment_root / "minimal_exp")

    cases = build_pcvr_experiment_cases(experiment_root)

    assert len(cases) == 1
    assert cases[0].path == "experiments/minimal_exp"
    assert cases[0].module == "experiments.minimal_exp"
    assert cases[0].name == "pcvr_minimal_exp"
    assert cases[0].model_class == "PCVRMinimalExp"
    assert cases[0].package_dir == (experiment_root / "minimal_exp").resolve()
