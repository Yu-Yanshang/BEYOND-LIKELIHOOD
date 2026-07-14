#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" dpo/scripts/prepare_ppo_data.py "$@"
