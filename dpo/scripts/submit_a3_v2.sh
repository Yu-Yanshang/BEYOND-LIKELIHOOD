#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_env.sh"

COMMAND="${1:-all}"
shift || true
export RUN_ID="${RUN_ID:-a3_v2_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${A3_V2_OUTPUT_ROOT:-${PROJECT_ROOT}/dpo/outputs/a3_v2_runs}"
LOG_DIR="${OUTPUT_ROOT}/${RUN_ID}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/a3_v2_${COMMAND}.log"
PID_FILE="${LOG_DIR}/a3_v2_${COMMAND}.pid"

cd "${PROJECT_ROOT}"
nohup bash "${SCRIPT_DIR}/run_a3_v2.sh" "${COMMAND}" "$@" > "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started A3 v2 ${COMMAND}."
echo "RUN_ID=${RUN_ID}"
echo "PID=$(cat "${PID_FILE}")"
echo "Log: ${LOG_FILE}"
echo "Tail: tail -f ${LOG_FILE}"
