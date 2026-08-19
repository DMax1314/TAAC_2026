from __future__ import annotations

from pathlib import Path


TEST_FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def fixture_path(*parts: str) -> Path:
    path = TEST_FIXTURES_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"test fixture not found: {path}")
    return path


def locate_repo_root(anchor: Path) -> Path:
    resolved = anchor.resolve()
    candidates = (resolved, *resolved.parents) if resolved.is_dir() else resolved.parents
    for parent in candidates:
        if (parent / "pyproject.toml").is_file() and (parent / "experiments").is_dir():
            return parent
    raise RuntimeError(f"could not locate TAAC_2026 repository root from {anchor}")
