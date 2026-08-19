"""Shared runtime state for application entrypoints."""

from __future__ import annotations

import os

def is_bundle_mode() -> bool:
    return os.environ.get("TAAC_BUNDLE_MODE") == "1"
