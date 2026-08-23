from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from taac2026.application.packaging.cli import build_training_bundle
from taac2026.infrastructure.io.json import loads
from tests.support.experiment_matrix import ExperimentCase, REPO_ROOT, discover_pcvr_experiment_cases


EXPERIMENT_CASES = discover_pcvr_experiment_cases()


def _code_package_names(code_package_path: Path) -> set[str]:
    with zipfile.ZipFile(code_package_path) as archive:
        return set(archive.namelist())


def _code_package_manifest(code_package_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(code_package_path) as archive:
        return loads(archive.read("project/.taac_training_manifest.json"))


@pytest.fixture(scope="module", params=EXPERIMENT_CASES, ids=lambda case: case.path)
def experiment_case(request) -> ExperimentCase:
    return request.param


@pytest.fixture(scope="module")
def built_bundle(experiment_case: ExperimentCase, tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp(f"bundle_{Path(experiment_case.path).name}")
    result = build_training_bundle(experiment_case.path, output_dir=output_dir, root=REPO_ROOT)
    return result, _code_package_manifest(result.code_package_path), _code_package_names(result.code_package_path)


def test_bundle_manifest_points_to_selected_experiment(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    result, manifest, _names = built_bundle

    assert result.output_dir.is_dir()
    assert result.run_script_path.exists()
    assert result.code_package_path.exists()
    assert manifest["bundle_format"] == "taac2026-training"
    assert manifest["bundled_experiment_path"] == experiment_case.path
    assert manifest["entrypoint"] == "run.sh"
    assert manifest["code_package"] == "code_package.zip"


def test_bundle_contains_model_and_typed_ns_config(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    _result, _manifest, names = built_bundle

    assert f"project/{experiment_case.path}/__init__.py" in names
    assert f"project/{experiment_case.path}/model.py" in names
    assert f"project/{experiment_case.path}/ns_groups.json" not in names
    assert "project/src/taac2026/application/training/cli.py" in names
    assert "project/src/taac2026/application/training/args.py" in names
    assert "project/src/taac2026/application/training/workflow.py" in names


def test_bundle_excludes_package_local_runtime_wrappers(
    built_bundle: tuple[object, dict[str, object], set[str]],
    experiment_case: ExperimentCase,
) -> None:
    _result, _manifest, names = built_bundle

    assert "project/run.sh" not in names
    assert f"project/{experiment_case.path}/run.sh" not in names
    assert f"project/{experiment_case.path}/train.py" not in names
    assert f"project/{experiment_case.path}/trainer.py" not in names
