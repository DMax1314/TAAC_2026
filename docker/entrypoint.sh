#!/usr/bin/env bash
set -euo pipefail

if [[ "${AUTO_SYNC:-1}" == "1" ]]; then
    uv sync --locked --extra "${UV_EXTRA}"
fi

exec "$@"
