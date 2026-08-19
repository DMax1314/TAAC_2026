from __future__ import annotations

import os
from pathlib import Path

import orjson


def main() -> None:
    payload = {
        "cwd": str(Path.cwd()),
        "experiment": os.environ.get("TAAC_EXPERIMENT"),
    }
    print(orjson.dumps(payload).decode())
