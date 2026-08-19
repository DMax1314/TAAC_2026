from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    print(json.dumps({"cwd": str(Path.cwd()), "argv": sys.argv[1:]}))


if __name__ == "__main__":
    main()
