from __future__ import annotations

import argparse
from importlib.resources import files
import json
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile

from ...infrastructure.io.console import print_summary_table
from ...infrastructure.io.files import ensure_dir


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "training_bundles"
DEFAULT_DATASET_ENV_VAR = "TAAC_DATASET_PATH"
DEFAULT_OUTPUT_ENV_VAR = "TAAC_OUTPUT_DIR"
DEFAULT_WORKDIR_ENV_VAR = "TAAC_BUNDLE_WORKDIR"
DEFAULT_CUDA_PROFILE_ENV_VAR = "TAAC_CUDA_PROFILE"
DEFAULT_ENABLE_TE_ENV_VAR = "TAAC_ENABLE_TE"
DEFAULT_FORCE_EXTRACT_ENV_VAR = "TAAC_FORCE_EXTRACT"
PAYLOAD_ARCHIVE_NAME = "runtime_payload.tar.gz"
PAYLOAD_PROJECT_DIRNAME = "project"
TEMPLATE_PACKAGE = "taac2026.application.maintenance.templates"

RUNTIME_TREE_RELATIVE_PATHS = (
    Path("src/taac2026/domain"),
    Path("src/taac2026/infrastructure"),
    Path("src/taac2026/application/training"),
)

RUNTIME_FILE_RELATIVE_PATHS = (
    Path("src/taac2026/__init__.py"),
    Path("src/taac2026/application/__init__.py"),
)


@dataclass(slots=True)
class TrainingBundleResult:
    output_path: Path
    bundle_name: str
    experiment_argument: str
    bundled_experiment_path: str
    payload_file_count: int
    payload_size_bytes: int
    archive_size_bytes: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle a TAAC 2026 experiment into a single zip for online training",
    )
    parser.add_argument("--experiment", required=True, help="Experiment package under config/, such as config/baseline")
    parser.add_argument(
        "--output",
        help="Zip file to create. Defaults to outputs/training_bundles/<bundle-name>.zip",
    )
    parser.add_argument(
        "--bundle-name",
        help="Override the generated bundle name used for the output filename and metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output zip if it already exists.",
    )
    return parser.parse_args(argv)


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return collapsed or "training-bundle"


def _normalize_experiment_relative_path(experiment_argument: str) -> Path:
    raw_value = experiment_argument.strip()
    if not raw_value:
        raise ValueError("--experiment must not be empty")

    candidate = Path(raw_value).expanduser()
    repo_candidate = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    if candidate.exists() or repo_candidate.exists():
        resolved = candidate.resolve() if candidate.exists() else repo_candidate.resolve()
        if resolved.is_file():
            if resolved.name != "__init__.py":
                raise ValueError("--experiment must point to a config package directory or its __init__.py")
            resolved = resolved.parent
        try:
            relative = resolved.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError("--experiment must resolve inside this repository") from exc
    else:
        module_parts = raw_value.split(".")
        if len(module_parts) < 2 or module_parts[0] != "config":
            raise ValueError("--experiment must be a config/ path or config.<name> module path")
        relative = Path(*module_parts)

    if not relative.parts or relative.parts[0] != "config":
        raise ValueError("--experiment must resolve under config/")

    experiment_dir = REPO_ROOT / relative
    if not experiment_dir.is_dir():
        raise ValueError(f"experiment package directory not found: {experiment_dir}")
    if not (experiment_dir / "__init__.py").exists():
        raise ValueError(f"experiment package is missing __init__.py: {experiment_dir}")
    return relative


def _read_template(template_name: str) -> str:
    return files(TEMPLATE_PACKAGE).joinpath(*template_name.split("/")).read_text(encoding="utf-8")


def _render_template(template_name: str, replacements: dict[str, str]) -> str:
    rendered = _read_template(template_name)
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    unresolved = re.findall(r"\{\{[a-zA-Z0-9_]+\}\}", rendered)
    if unresolved:
        placeholder_list = ", ".join(sorted(set(unresolved)))
        raise ValueError(f"Template {template_name} has unresolved placeholders: {placeholder_list}")
    return rendered


def _render_run_sh(bundled_experiment_path: str) -> str:
    return _render_template(
        "run.sh.tmpl",
        {
            "payload_archive_name": PAYLOAD_ARCHIVE_NAME,
            "payload_project_dirname": PAYLOAD_PROJECT_DIRNAME,
            "dataset_env_var": DEFAULT_DATASET_ENV_VAR,
            "workdir_expr": f"${{{DEFAULT_WORKDIR_ENV_VAR}:-$SCRIPT_DIR/runtime}}",
            "dataset_path_expr": f"${{{DEFAULT_DATASET_ENV_VAR}:-}}",
            "output_dir_expr": f"${{{DEFAULT_OUTPUT_ENV_VAR}:-$SCRIPT_DIR/outputs}}",
            "cuda_profile_expr": f"${{{DEFAULT_CUDA_PROFILE_ENV_VAR}:-cuda126}}",
            "enable_te_expr": f"${{{DEFAULT_ENABLE_TE_ENV_VAR}:-0}}",
            "force_extract_expr": f"${{{DEFAULT_FORCE_EXTRACT_ENV_VAR}:-0}}",
            "bundled_experiment_path": bundled_experiment_path,
        },
    )


def _render_readme(bundle_name: str, bundled_experiment_path: str) -> str:
    return _render_template(
        "bundle_readme.md.tmpl",
        {
            "bundle_name": bundle_name,
            "payload_archive_name": PAYLOAD_ARCHIVE_NAME,
            "dataset_env_var": DEFAULT_DATASET_ENV_VAR,
            "output_env_var": DEFAULT_OUTPUT_ENV_VAR,
            "workdir_env_var": DEFAULT_WORKDIR_ENV_VAR,
            "cuda_profile_env_var": DEFAULT_CUDA_PROFILE_ENV_VAR,
            "enable_te_env_var": DEFAULT_ENABLE_TE_ENV_VAR,
            "force_extract_env_var": DEFAULT_FORCE_EXTRACT_ENV_VAR,
            "dataset_env_reference": f"${{{DEFAULT_DATASET_ENV_VAR}}}",
            "output_env_reference": f"${{{DEFAULT_OUTPUT_ENV_VAR}}}",
            "bundled_experiment_path": bundled_experiment_path,
        },
    )


def _ignore_copy_junk(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))}


def _copy_relative_path(relative_path: Path, destination_root: Path) -> None:
    source = REPO_ROOT / relative_path
    destination = destination_root / relative_path
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=_ignore_copy_junk)
        return
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)


def _count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def _count_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_payload_archive(payload_root: Path, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with tarfile.open(output_path, mode="w:gz") as archive:
        archive.add(payload_root, arcname=PAYLOAD_PROJECT_DIRNAME)


def _write_zip(bundle_root: Path, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, arcname=path.relative_to(bundle_root).as_posix())


def build_training_bundle(
    experiment_argument: str,
    *,
    output_path: str | Path | None = None,
    bundle_name: str | None = None,
    force: bool = False,
) -> TrainingBundleResult:
    experiment_relative_path = _normalize_experiment_relative_path(experiment_argument)
    normalized_bundle_name = _slugify(bundle_name or f"{experiment_relative_path.name}-train-bundle")
    target_path = Path(output_path).expanduser() if output_path is not None else DEFAULT_OUTPUT_ROOT / f"{normalized_bundle_name}.zip"
    archive_path = target_path if target_path.is_absolute() else (REPO_ROOT / target_path)
    archive_path = archive_path.resolve()

    if archive_path.exists():
        if archive_path.is_dir():
            raise IsADirectoryError(f"output path is a directory: {archive_path}")
        if not force:
            raise FileExistsError(f"output zip already exists: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="taac_train_bundle_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        payload_root = temp_dir / PAYLOAD_PROJECT_DIRNAME
        bundle_root = temp_dir / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)

        for relative_path in RUNTIME_TREE_RELATIVE_PATHS:
            _copy_relative_path(relative_path, payload_root)
        for relative_path in RUNTIME_FILE_RELATIVE_PATHS:
            _copy_relative_path(relative_path, payload_root)
        _copy_relative_path(experiment_relative_path, payload_root)
        _copy_relative_path(Path("pyproject.toml"), payload_root)
        _copy_relative_path(Path("uv.lock"), payload_root)
        _copy_relative_path(Path("README.md"), payload_root)

        payload_archive = bundle_root / PAYLOAD_ARCHIVE_NAME
        _write_payload_archive(payload_root, payload_archive)

        bundled_experiment_path = experiment_relative_path.as_posix()
        run_sh_path = bundle_root / "run.sh"
        _write_text(run_sh_path, _render_run_sh(bundled_experiment_path))
        run_sh_path.chmod(0o755)
        _write_text(bundle_root / "README.md", _render_readme(normalized_bundle_name, bundled_experiment_path))

        payload_file_count = _count_files(payload_root)
        payload_size_bytes = _count_bytes(payload_root)
        manifest = {
            "schema_version": 1,
            "bundle_name": normalized_bundle_name,
            "experiment_argument": experiment_argument,
            "bundled_experiment_path": bundled_experiment_path,
            "payload_archive": PAYLOAD_ARCHIVE_NAME,
            "payload_root": PAYLOAD_PROJECT_DIRNAME,
            "entrypoint": "run.sh",
            "lockfile": "uv.lock",
            "runtime_env": {
                "dataset_path": DEFAULT_DATASET_ENV_VAR,
                "output_dir": DEFAULT_OUTPUT_ENV_VAR,
                "workdir": DEFAULT_WORKDIR_ENV_VAR,
                "cuda_profile": DEFAULT_CUDA_PROFILE_ENV_VAR,
                "enable_te": DEFAULT_ENABLE_TE_ENV_VAR,
                "force_extract": DEFAULT_FORCE_EXTRACT_ENV_VAR,
            },
            "payload_stats": {
                "file_count": payload_file_count,
                "size_bytes": payload_size_bytes,
            },
        }
        _write_json(bundle_root / "bundle_manifest.json", manifest)

        temp_archive_path = temp_dir / "bundle.zip"
        _write_zip(bundle_root, temp_archive_path)
        ensure_dir(archive_path.parent)
        if archive_path.exists():
            archive_path.unlink()
        shutil.move(str(temp_archive_path), str(archive_path))

    return TrainingBundleResult(
        output_path=archive_path,
        bundle_name=normalized_bundle_name,
        experiment_argument=experiment_argument,
        bundled_experiment_path=bundled_experiment_path,
        payload_file_count=payload_file_count,
        payload_size_bytes=payload_size_bytes,
        archive_size_bytes=archive_path.stat().st_size,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_training_bundle(
            args.experiment,
            output_path=args.output,
            bundle_name=args.bundle_name,
            force=args.force,
        )
    except (FileExistsError, IsADirectoryError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print_summary_table(
        "taac-package-train",
        [
            ("bundle_name", result.bundle_name),
            ("experiment", result.experiment_argument),
            ("bundled_experiment", result.bundled_experiment_path),
            ("payload_files", result.payload_file_count),
            ("payload_size_bytes", result.payload_size_bytes),
            ("archive_size_bytes", result.archive_size_bytes),
            ("output", result.output_path),
        ],
    )
    return 0


__all__ = ["TrainingBundleResult", "build_training_bundle", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())