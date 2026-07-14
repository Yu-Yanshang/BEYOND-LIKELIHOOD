#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-a2full_dpo_lora_ppo_20260706}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"

ROOT_OUT="${PROJECT_ROOT}/dpo/outputs/${RUN_ID}"
LOG_DIR="${ROOT_OUT}/logs"
mkdir -p "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/epoch_continue_master.log"
PID_FILE="${ROOT_OUT}/epoch_continue_submit.pid"

cd "${PROJECT_ROOT}"
nohup bash -lc "RUN_ID='${RUN_ID}' GPU_ID='${GPU_ID}' PYTHON_BIN='${PYTHON_BIN}' bash dpo/scripts/run_a2full_epoch_continue_sequence.sh" >"${MASTER_LOG}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"

echo "pid=${PID}"
echo "log=${MASTER_LOG}"
echo "pid_file=${PID_FILE}"
