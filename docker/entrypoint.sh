#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [[ ! -f pyproject.toml ]]; then
    echo "pyproject.toml not found under /workspace" >&2
    exit 2
fi

actual_uv_extra="${UV_EXTRA:-}"
expected_uv_extra="${EXPECTED_UV_EXTRA:-cuda126}"
if [[ "${actual_uv_extra}" != "${expected_uv_extra}" ]]; then
    echo "Unsupported UV_EXTRA: expected ${expected_uv_extra}, got ${actual_uv_extra:-<unset>}" >&2
    exit 64
fi

actual_python_version="$("${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
expected_python_version="${EXPECTED_PYTHON_VERSION:-3.10.20}"
if [[ "${actual_python_version}" != "${expected_python_version}" ]]; then
    echo "Unsupported Python version: expected ${expected_python_version}, got ${actual_python_version}" >&2
    exit 65
fi

actual_base_image="${BASE_IMAGE:-${IMAGE_BASE_NAME:-}}"
expected_base_image="${EXPECTED_BASE_IMAGE:-nvidia/cuda:12.6.0-devel-ubuntu22.04}"
if [[ -n "${actual_base_image}" && "${actual_base_image}" != "${expected_base_image}" ]]; then
    echo "Unsupported base image: expected ${expected_base_image}, got ${actual_base_image}" >&2
    exit 66
fi

if command -v nvcc >/dev/null 2>&1; then
    cuda_version="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1 | tr -d '[:space:]')"
    if [[ -n "${cuda_version}" && "${cuda_version}" != 12.6* ]]; then
        echo "Unsupported CUDA toolkit: expected 12.6.x, got ${cuda_version}" >&2
        exit 67
    fi
fi

if [[ "${AUTO_SYNC:-1}" == "1" ]]; then
    sync_args=(--locked --extra "${actual_uv_extra}")
    uv sync "${sync_args[@]}"
fi

exec "$@"
