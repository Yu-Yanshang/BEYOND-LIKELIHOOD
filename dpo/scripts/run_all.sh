#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"

cd "${PROJECT_ROOT}"

bash "dpo/scripts/prepare_data.sh"
bash "dpo/scripts/train.sh"
"${PYTHON_BIN}" "dpo/scripts/test_code_dpo.py"
