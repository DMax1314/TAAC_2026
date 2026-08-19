from __future__ import annotations

import os
import sys
from pathlib import Path

import orjson


def main() -> None:
    payload = {
        "cwd": str(Path.cwd()),
        "argv": sys.argv[1:],
        "experiment": os.environ.get("TAAC_EXPERIMENT"),
    }
    print(orjson.dumps(payload).decode())


if __name__ == "__main__":
    main()
