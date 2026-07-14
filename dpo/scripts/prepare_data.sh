#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
SOURCE_DIR="${SOURCE_DIR:-py-dpo-v0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-dpo/data}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "dpo/scripts/prepare_dpo_data.py" \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test_size "${TEST_SIZE:-500}" \
  --seed "${SEED:-42}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-0}"
