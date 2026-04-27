from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from taac2026.application.maintenance.package_training import build_training_bundle


def _read_payload_names(bundle_path: Path) -> set[str]:
    with zipfile.ZipFile(bundle_path) as bundle_archive:
        payload_bytes = bundle_archive.read("runtime_payload.tar.gz")
    with tarfile.open(fileobj=BytesIO(payload_bytes), mode="r:gz") as payload_archive:
        return {member.name for member in payload_archive.getmembers()}


def test_build_training_bundle_writes_single_zip_with_expected_layout(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline_bundle.zip"

    result = build_training_bundle("config/baseline", output_path=output_path)

    assert result.output_path == output_path.resolve()
    assert output_path.exists()

    with zipfile.ZipFile(output_path) as bundle_archive:
        names = set(bundle_archive.namelist())
        assert {"README.md", "bundle_manifest.json", "run.sh", "runtime_payload.tar.gz"} <= names
        run_sh = bundle_archive.read("run.sh").decode("utf-8")
        manifest = json.loads(bundle_archive.read("bundle_manifest.json").decode("utf-8"))

    assert "TAAC_DATASET_PATH" in run_sh
    assert 'CUDA_PROFILE="${TAAC_CUDA_PROFILE:-cuda126}"' in run_sh
    assert 'uv sync --locked --extra "$CUDA_PROFILE" "${UV_SYNC_EXTRA_ARGS[@]}"' in run_sh
    assert "Unsupported CUDA profile" in run_sh
    assert 'uv run taac-train --experiment "./config/baseline"' in run_sh
    assert manifest["bundled_experiment_path"] == "config/baseline"
    assert manifest["lockfile"] == "uv.lock"
    assert manifest["runtime_env"]["cuda_profile"] == "TAAC_CUDA_PROFILE"


def test_build_training_bundle_payload_is_trimmed_to_training_runtime(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline_bundle.zip"
    build_training_bundle("config.baseline", output_path=output_path)

    payload_names = _read_payload_names(output_path)

    assert "project/config/baseline/__init__.py" in payload_names
    assert "project/src/taac2026/application/training/cli.py" in payload_names
    assert "project/src/taac2026/infrastructure/io/files.py" in payload_names
    assert "project/uv.lock" in payload_names
    assert "project/src/taac2026/application/search/cli.py" not in payload_names
    assert "project/tests/unit/test_package_training.py" not in payload_names
    assert "project/docs/getting-started.md" not in payload_names


def test_build_training_bundle_copies_root_pyproject_and_readme(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline_bundle.zip"
    build_training_bundle("config/baseline", output_path=output_path)

    with zipfile.ZipFile(output_path) as bundle_archive:
        payload_bytes = bundle_archive.read("runtime_payload.tar.gz")
    with tarfile.open(fileobj=BytesIO(payload_bytes), mode="r:gz") as payload_archive:
        pyproject_member = payload_archive.extractfile("project/pyproject.toml")
        assert pyproject_member is not None
        pyproject_text = pyproject_member.read().decode("utf-8")

        lockfile_member = payload_archive.extractfile("project/uv.lock")
        assert lockfile_member is not None
        payload_lockfile_text = lockfile_member.read().decode("utf-8")

        readme_member = payload_archive.extractfile("project/README.md")
        assert readme_member is not None
        payload_readme_text = readme_member.read().decode("utf-8")

    source_pyproject_text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    source_lockfile_text = (Path(__file__).resolve().parents[2] / "uv.lock").read_text(encoding="utf-8")
    source_readme_text = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")

    assert pyproject_text == source_pyproject_text
    assert payload_lockfile_text == source_lockfile_text
    assert payload_readme_text == source_readme_text


def test_build_training_bundle_refuses_to_overwrite_existing_output_without_force(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline_bundle.zip"
    build_training_bundle("config/baseline", output_path=output_path)

    with pytest.raises(FileExistsError, match="output zip already exists"):
        build_training_bundle("config/baseline", output_path=output_path)


def test_build_training_bundle_force_replaces_existing_output_file(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline_bundle.zip"
    output_path.write_text("stale", encoding="utf-8")

    result = build_training_bundle("config/baseline", output_path=output_path, force=True)

    assert result.output_path == output_path.resolve()
    with zipfile.ZipFile(output_path) as bundle_archive:
        assert "run.sh" in bundle_archive.namelist()


def test_build_training_bundle_rejects_directory_output_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle_dir"
    output_dir.mkdir()

    with pytest.raises(IsADirectoryError, match="output path is a directory"):
        build_training_bundle("config/baseline", output_path=output_dir, force=True)