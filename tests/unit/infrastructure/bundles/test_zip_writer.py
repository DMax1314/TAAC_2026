from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from taac2026.infrastructure.bundles.manifest_store import build_bundle_manifest
from taac2026.infrastructure.bundles.zip_writer import write_workspace_code_package
from tests.support.paths import fixture_path


def test_write_workspace_code_package_places_external_experiment_under_manifest_path(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    (root / "experiments").mkdir(parents=True)
    (root / "experiments" / "__init__.py").touch()
    (root / "src" / "taac2026").mkdir(parents=True)
    (root / "src" / "taac2026" / "__init__.py").touch()
    (root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    external_experiment = tmp_path / "custom_exp"
    shutil.copytree(fixture_path("experiments", "module_loader_package"), external_experiment)
    manifest = build_bundle_manifest(kind="training", experiment_path=external_experiment, root=root)
    code_package_path = tmp_path / "code_package.zip"

    write_workspace_code_package(
        code_package_path=code_package_path,
        experiment_path=external_experiment,
        root=root,
        manifest=manifest,
        manifest_name=".taac_training_manifest.json",
    )

    bundled_path = str(manifest["bundled_experiment_path"])
    with zipfile.ZipFile(code_package_path) as archive:
        names = set(archive.namelist())
    assert f"project/{bundled_path}/__init__.py" in names
    assert f"project/{bundled_path}/model.py" in names
    assert "project/experiments/__init__.py" in names
