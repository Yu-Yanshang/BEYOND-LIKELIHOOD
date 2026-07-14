#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" sft/scripts/infer.py "$@"
