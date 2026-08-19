from __future__ import annotations

import os
import sys
from pathlib import Path

import orjson


Path(os.environ["TAAC_TEST_PIP_ARGS_PATH"]).write_bytes(orjson.dumps(sys.argv[1:]))
