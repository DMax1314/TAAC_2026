#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

REPO_ROOT="$DEFAULT_REPO_ROOT"
REQUESTED_PROFILE="${TAAC_REQUESTED_PROFILE:-${TAAC_CUDA_PROFILE:-}}"
REQUESTED_PYTHON="${TAAC_REQUESTED_PYTHON:-}"
UV_INSTALL_URL="${TAAC_UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
PYPI_INDEX_URL="${TAAC_PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_CPU_INDEX_URL="${TAAC_PYTORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
PYTORCH_CUDA126_INDEX_URL="${TAAC_PYTORCH_CUDA126_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
PYTORCH_CUDA128_INDEX_URL="${TAAC_PYTORCH_CUDA128_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTORCH_CUDA130_INDEX_URL="${TAAC_PYTORCH_CUDA130_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
OUTPUT_PATH=""

usage() {
    cat <<'EOF'
Usage: bash tools/log_host_device_info.sh [options]

Options:
  --repo-root PATH           Override repo root shown in the log.
  --requested-profile NAME   Record the requested runtime profile.
  --requested-python VER     Record the requested Python version.
    --uv-install-url URL       Override the uv installer URL probe target.
  --output PATH              Tee the log to PATH while printing to stdout.
  -h, --help                 Show this help message.

Examples:
  bash tools/log_host_device_info.sh --requested-profile cpu --requested-python 3.13
  bash tools/log_host_device_info.sh --output /tmp/host-device.log
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo-root)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                REPO_ROOT="$2"
                shift 2
                ;;
            --requested-profile)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                REQUESTED_PROFILE="$2"
                shift 2
                ;;
            --requested-python)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                REQUESTED_PYTHON="$2"
                shift 2
                ;;
            --output)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                OUTPUT_PATH="$2"
                shift 2
                ;;
            --uv-install-url)
                [[ $# -ge 2 ]] || {
                    echo "Missing value for $1" >&2
                    exit 2
                }
                UV_INSTALL_URL="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1" >&2
                usage >&2
                exit 2
                ;;
        esac
    done
}

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

log_line() {
    printf '[%s] %s\n' "$(timestamp)" "$*"
}

start_capture() {
    if [[ -z "$OUTPUT_PATH" ]]; then
        return
    fi
    mkdir -p "$(dirname "$OUTPUT_PATH")"
    exec > >(tee -a "$OUTPUT_PATH") 2>&1
    log_line "device log: $OUTPUT_PATH"
}

run_logged_command() {
    local title="$1"
    shift
    local command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        log_line "---- $title ----"
        "$@" || log_line "$title failed with exit code $?"
    else
        log_line "---- $title unavailable: $command_name not found ----"
    fi
}

log_os_release() {
    if [[ -r /etc/os-release ]]; then
        log_line "---- os-release ----"
        sed -n 's/^PRETTY_NAME=//p; s/^VERSION=//p' /etc/os-release | tr -d '"'
    else
        log_line "---- os-release unavailable ----"
    fi
}

log_network_info() {
    if command -v ip >/dev/null 2>&1; then
        log_line "---- network ----"
        ip -br addr || log_line "network failed with exit code $?"
    else
        log_line "---- network unavailable: ip not found ----"
    fi
}

log_nvidia_device_nodes() {
    if compgen -G "/dev/nvidia*" >/dev/null; then
        log_line "---- nvidia device nodes ----"
        ls -l /dev/nvidia* || true
    else
        log_line "nvidia device nodes: none"
    fi
}

log_dri_nodes() {
    if [[ -e /dev/dri ]]; then
        log_line "---- /dev/dri ----"
        ls -l /dev/dri || true
    else
        log_line "/dev/dri: none"
    fi
}

log_python_info() {
    if command -v python3 >/dev/null 2>&1; then
        log_line "---- python3 ----"
        python3 --version
        python3 - <<'PY'
import os
import platform
import sys

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
print(f"nvidia_visible_devices={os.environ.get('NVIDIA_VISIBLE_DEVICES', '<unset>')}")
PY
    else
        log_line "---- python3 unavailable: python3 not found ----"
    fi
}

log_python_packages() {
    if ! command -v python3 >/dev/null 2>&1; then
        log_line "---- python packages unavailable: python3 not found ----"
        return
    fi

    log_line "---- python packages ----"
    python3 - <<'PY'
from importlib import metadata

packages = []
for dist in metadata.distributions():
    name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.metadata.get("name") or dist.name
    version = dist.version
    packages.append((str(name), str(version)))

packages.sort(key=lambda item: item[0].lower())
print(f"installed_python_packages={len(packages)}")
for name, version in packages:
    print(f"{name}=={version}")
PY
}

log_url_probe() {
    local label="$1"
    local url="$2"

    log_line "${label}_url=$url"
    if command -v curl >/dev/null 2>&1; then
        log_line "${label}_probe_tool=curl"
        local http_code
        http_code="$(curl -I -L -sS --max-time 10 -o /dev/null -w '%{http_code}' "$url" || true)"
        if [[ -n "$http_code" && "$http_code" != "000" ]]; then
            log_line "${label}_probe=reachable"
            log_line "${label}_http_code=$http_code"
        else
            log_line "${label}_probe=failed"
        fi
        return
    fi

    if command -v wget >/dev/null 2>&1; then
        log_line "${label}_probe_tool=wget"
        if wget --spider -q --timeout=10 --tries=1 "$url" >/dev/null 2>&1; then
            log_line "${label}_probe=reachable"
        else
            log_line "${label}_probe=failed"
        fi
        return
    fi

    log_line "${label}_probe_tool=none"
    log_line "${label}_probe=unavailable"
}

pytorch_index_url_for_profile() {
    case "$1" in
        cpu)
            printf '%s' "$PYTORCH_CPU_INDEX_URL"
            ;;
        cuda126)
            printf '%s' "$PYTORCH_CUDA126_INDEX_URL"
            ;;
        cuda128)
            printf '%s' "$PYTORCH_CUDA128_INDEX_URL"
            ;;
        cuda130)
            printf '%s' "$PYTORCH_CUDA130_INDEX_URL"
            ;;
        *)
            return 1
            ;;
    esac
}

log_uv_bootstrap_status() {
    log_line "---- uv bootstrap ----"
    log_line "uv_install_url=$UV_INSTALL_URL"
    if command -v uv >/dev/null 2>&1; then
        log_line "uv_present=1"
        uv --version
    else
        log_line "uv_present=0"
    fi

    log_url_probe "uv_download" "$UV_INSTALL_URL"
}

log_dependency_index_status() {
    log_line "---- dependency indexes ----"
    log_url_probe "pypi_index" "$PYPI_INDEX_URL"

    if [[ -n "$REQUESTED_PROFILE" ]]; then
        local requested_url
        if requested_url="$(pytorch_index_url_for_profile "$REQUESTED_PROFILE")"; then
            log_line "pytorch_probe_profile=$REQUESTED_PROFILE"
            log_url_probe "pytorch_index_${REQUESTED_PROFILE}" "$requested_url"
        else
            log_line "pytorch_probe_profile=$REQUESTED_PROFILE"
            log_line "pytorch_probe=unsupported-profile"
        fi
        return
    fi

    log_line "pytorch_probe_profile=all"
    local profile
    local profile_url
    for profile in cpu cuda126 cuda128 cuda130; do
        profile_url="$(pytorch_index_url_for_profile "$profile")"
        log_url_probe "pytorch_index_${profile}" "$profile_url"
    done
}

log_build_tools() {
    log_line "---- build tools ----"
    local tool
    for tool in gcc g++ make cmake ninja pkg-config cc c++; do
        if command -v "$tool" >/dev/null 2>&1; then
            log_line "$tool=present"
            "$tool" --version || log_line "$tool --version failed with exit code $?"
        else
            log_line "$tool=missing"
        fi
    done
}

main() {
    parse_args "$@"
    start_capture

    log_line "==== Host and device information ===="
    log_line "repo_root=$REPO_ROOT"
    if [[ -n "$REQUESTED_PROFILE" ]]; then
        log_line "requested_profile=$REQUESTED_PROFILE"
    fi
    if [[ -n "$REQUESTED_PYTHON" ]]; then
        log_line "requested_python=$REQUESTED_PYTHON"
    fi
    log_line "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    log_line "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"

    log_os_release
    run_logged_command "hostname" hostname
    run_logged_command "uptime" uptime
    run_logged_command "kernel" uname -a
    run_logged_command "cpu" lscpu
    run_logged_command "memory" free -h
    run_logged_command "block devices" lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL
    run_logged_command "disk usage" df -h "$REPO_ROOT" /tmp
    log_network_info
    log_nvidia_device_nodes
    log_dri_nodes
    run_logged_command "nvidia-smi list" nvidia-smi -L
    run_logged_command "nvidia-smi" nvidia-smi
    run_logged_command "nvcc" nvcc --version
    run_logged_command "uv" uv --version
    log_uv_bootstrap_status
    log_dependency_index_status
    log_build_tools
    log_python_info
    log_python_packages
}

main "$@"