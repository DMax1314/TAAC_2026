"""Bundle manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Literal

from pydantic import Field

from taac2026 import __version__
from taac2026.infrastructure.io.files import stable_hash64
from taac2026.domain.validation import TAACBoundaryModel


BundleKind = Literal["training", "inference"]

TRAINING_BUNDLE_FORMAT = "taac2026-training"
INFERENCE_BUNDLE_FORMAT = "taac2026-inference"

@dataclass(frozen=True, slots=True)
class BundleDefinition:
    kind: BundleKind
    manifest_name: str
    entrypoint: str
    bundle_format: str
    runtime_env: tuple[tuple[str, str], ...]
    summary_runtime_fields: tuple[tuple[str, str], ...]

    def runtime_env_dict(self) -> dict[str, str]:
        return dict(self.runtime_env)

    @property
    def runtime_env_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.runtime_env)


_BUNDLE_DEFINITIONS: dict[BundleKind, BundleDefinition] = {
    "training": BundleDefinition(
        kind="training",
        manifest_name=".taac_training_manifest.json",
        entrypoint="run.sh",
        bundle_format=TRAINING_BUNDLE_FORMAT,
        runtime_env=(
            ("dataset_path", "TRAIN_DATA_PATH"),
            ("schema_path", "TAAC_SCHEMA_PATH"),
            ("checkpoint_path", "TRAIN_CKPT_PATH"),
            ("cuda_profile", "TAAC_CUDA_PROFILE"),
            ("pip_extras", "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)"),
        ),
        summary_runtime_fields=(
            ("dataset", "dataset_path"),
            ("schema", "schema_path"),
            ("output", "checkpoint_path"),
            ("cuda profile", "cuda_profile"),
            ("pip extras", "pip_extras"),
        ),
    ),
    "inference": BundleDefinition(
        kind="inference",
        manifest_name=".taac_inference_manifest.json",
        entrypoint="infer.py",
        bundle_format=INFERENCE_BUNDLE_FORMAT,
        runtime_env=(
            ("model_path", "MODEL_OUTPUT_PATH"),
            ("dataset_path", "EVAL_DATA_PATH"),
            ("result_path", "EVAL_RESULT_PATH"),
            ("schema_path", "TAAC_SCHEMA_PATH"),
            ("pip_extras", "TAAC_BUNDLE_PIP_EXTRAS (optional; defaults to runtime-only install with no dev extra)"),
        ),
        summary_runtime_fields=(
            ("model", "model_path"),
            ("dataset", "dataset_path"),
            ("result", "result_path"),
            ("schema", "schema_path"),
            ("pip extras", "pip_extras"),
        ),
    ),
}


def get_bundle_definition(kind: BundleKind) -> BundleDefinition:
    return _BUNDLE_DEFINITIONS[kind]


def _safe_experiment_archive_name(experiment_path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", experiment_path.name).strip("_") or "external_experiment"
    suffix = f"{stable_hash64(str(experiment_path.expanduser().resolve())):016x}"[:12]
    return f"{name}_{suffix}"


def _format_generated_site_experiment_path_error(*, relative_path: str, experiment_path: Path, root: Path) -> str:
    return (
        "bundle experiment path resolved inside generated site/ output; "
        f"got {relative_path!r} from resolved experiment path {experiment_path} "
        f"relative to workspace root {root}. "
        "Pass an experiment package such as 'experiments/baseline' from the repository root, "
        "or pass an explicit external experiment package path."
    )


def _bundled_experiment_path(experiment_path: Path, root: Path) -> str:
    try:
        relative_path = experiment_path.relative_to(root).as_posix()
    except ValueError:
        return f"experiments/{_safe_experiment_archive_name(experiment_path)}"
    if relative_path.startswith("experiments/"):
        return relative_path
    if relative_path.startswith("site/"):
        raise ValueError(
            _format_generated_site_experiment_path_error(
                relative_path=relative_path,
                experiment_path=experiment_path,
                root=root,
            )
        )
    return f"experiments/{_safe_experiment_archive_name(experiment_path)}"


def _validate_bundled_experiment_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "experiments":
        raise ValueError(
            "bundle manifest must contain a relative bundled_experiment_path under 'experiments/'; "
            f"got {value!r}"
        )


class FrameworkMetadata(TAACBoundaryModel):
    name: str = "taac2026"
    version: str


class BundleManifest(TAACBoundaryModel):
    kind: BundleKind = Field(alias="bundle_kind")
    bundle_format: str
    framework: FrameworkMetadata
    bundled_experiment_path: str
    entrypoint: str
    code_package: str
    runtime_env: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return self.model_dump(mode="python", by_alias=True)


def build_bundle_manifest(*, kind: BundleKind, experiment_path: Path, root: Path) -> dict[str, object]:
    definition = get_bundle_definition(kind)
    manifest = BundleManifest(
        bundle_kind=kind,
        bundle_format=definition.bundle_format,
        framework=FrameworkMetadata(version=__version__),
        bundled_experiment_path=_bundled_experiment_path(experiment_path, root),
        entrypoint=definition.entrypoint,
        code_package="code_package.zip",
        runtime_env=definition.runtime_env_dict(),
    )
    return validate_bundle_manifest(manifest.to_dict(), kind=kind)


def validate_bundle_manifest(manifest: Mapping[str, Any], *, kind: BundleKind | None = None) -> dict[str, object]:
    model = BundleManifest.model_validate(dict(manifest))
    payload = model.to_dict()

    manifest_kind = model.kind
    if kind is not None and manifest_kind != kind:
        raise ValueError(f"bundle manifest kind mismatch: expected {kind}, got {manifest_kind}")

    definition = get_bundle_definition(manifest_kind)
    if model.bundle_format != definition.bundle_format:
        raise ValueError(f"unsupported {manifest_kind} bundle format: {model.bundle_format}")

    _validate_bundled_experiment_path(model.bundled_experiment_path)
    if model.entrypoint != definition.entrypoint:
        raise ValueError(f"invalid {manifest_kind} bundle entrypoint: {model.entrypoint}")
    if model.code_package != "code_package.zip":
        raise ValueError(f"invalid bundle code_package: {model.code_package}")

    missing_env = sorted(set(definition.runtime_env_keys) - set(model.runtime_env))
    if missing_env:
        raise KeyError(f"bundle manifest runtime_env is missing key(s): {', '.join(missing_env)}")

    return payload


__all__ = [
    "INFERENCE_BUNDLE_FORMAT",
    "TRAINING_BUNDLE_FORMAT",
    "BundleDefinition",
    "BundleKind",
    "BundleManifest",
    "FrameworkMetadata",
    "build_bundle_manifest",
    "get_bundle_definition",
    "validate_bundle_manifest",
]
