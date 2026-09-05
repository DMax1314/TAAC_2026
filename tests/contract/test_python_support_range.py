"""Cross-file consistency guards for the declared Python support range.

Keeps `requires-python`, ruff `target-version`, the CI test matrix, README
badge and docs version statements in sync. Any version declaration drift
fails here before it can reach reviewers or users.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.specifiers import SpecifierSet

from tests.support.paths import locate_repo_root

REPO_ROOT = locate_repo_root(Path(__file__))

# Text patterns that embed the declared support range.
_README_BADGE_PATTERN = re.compile(r"Python-(\d+\.\d+)--(\d+\.\d+)")


def _pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _ci_workflow() -> dict:
    with (REPO_ROOT / ".github/workflows/ci.yml").open() as fh:
        return yaml.safe_load(fh)


def _docs_workflow() -> dict:
    with (REPO_ROOT / ".github/workflows/deploy-docs.yml").open() as fh:
        return yaml.safe_load(fh)


def _support_spec() -> SpecifierSet:
    return SpecifierSet(_pyproject()["project"]["requires-python"])


def _support_bounds() -> tuple[str, str]:
    """Return (min_version, max_supported_minor) derived from requires-python.

    Assumes the conventional ``>=X.Y,<X.Z`` declaration shape.
    """
    spec = _support_spec()
    lower = next((s.version for s in spec if s.operator == ">="), None)
    upper = next((s.version for s in spec if s.operator == "<"), None)
    if lower is None or upper is None:
        pytest.fail(f"unsupported requires-python shape: {spec!s}")
    major, minor = upper.split(".")[:2]
    return lower, f"{major}.{int(minor) - 1}"


def test_ruff_target_version_matches_requires_python() -> None:
    min_version, _ = _support_bounds()
    expected = "py" + min_version.replace(".", "")
    actual = _pyproject()["tool"]["ruff"]["target-version"]
    assert actual == expected, (
        f"ruff target-version {actual!r} drifted from requires-python "
        f"({expected!r} expected); update [tool.ruff] target-version"
    )


def test_ci_matrix_within_requires_python() -> None:
    spec = _support_spec()
    versions = _ci_workflow()["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    assert versions, "CI test matrix must not be empty"
    for version in versions:
        assert spec.contains(version), (
            f"CI matrix Python {version} is outside requires-python {spec!s}; "
            "update the test matrix or the support range"
        )


def test_readme_badge_matches_requires_python() -> None:
    min_version, max_version = _support_bounds()
    text = (REPO_ROOT / "README.md").read_text()
    match = _README_BADGE_PATTERN.search(text)
    assert match is not None, "README Python badge missing"
    assert (match.group(1), match.group(2)) == (min_version, max_version), (
        f"README badge Python-{match.group(1)}--{match.group(2)} drifted from "
        f"requires-python ({min_version}--{max_version} expected)"
    )


def test_docs_version_statements_match_requires_python() -> None:
    min_version, max_version = _support_bounds()
    getting_started = (REPO_ROOT / "docs/getting-started.md").read_text()
    testing = (REPO_ROOT / "docs/guide/testing.md").read_text()
    assert f"Python {min_version} - {max_version}" in getting_started
    assert f"Python {min_version} 到 {max_version}" in testing


def test_local_python_version_within_requires_python() -> None:
    spec = _support_spec()
    pin = (REPO_ROOT / ".python-version").read_text().strip()
    assert spec.contains(pin), (
        f".python-version pins Python {pin}, outside requires-python {spec!s}"
    )


def test_docs_workflow_python_within_requires_python() -> None:
    version = _docs_workflow()["env"]["DOCS_PYTHON_VERSION"]
    assert _support_spec().contains(version), (
        f"deploy-docs.yml pins Python {version}, outside requires-python "
        f"{_support_spec()!s}"
    )
