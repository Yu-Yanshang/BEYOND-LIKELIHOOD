#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMMAND="${1:-acceptance}"
if [[ "${COMMAND}" != "acceptance" && "${COMMAND}" != "formal" && "${COMMAND}" != "all" ]]; then
  echo "Usage: RUN_ID=<id> A3_RUN_ID=<id> $0 {acceptance|formal}" >&2
  exit 2
fi
RUN_ID="${RUN_ID:-a2_v2_$(date +%Y%m%d_%H%M%S)}"
PROFILE="acceptance"; [[ "${COMMAND}" != "acceptance" ]] && PROFILE="formal"
export RUN_ID PROFILE

# Prepare synchronously so the master log can live inside the protected run directory.
RUN_ID="${RUN_ID}" PROFILE="${PROFILE}" bash "${SCRIPT_DIR}/run_a2_v2.sh" prepare
LOG_DIR="${PROJECT_ROOT}/sft/outputs/a2_v2_runs/${RUN_ID}/logs"
LOG_FILE="${LOG_DIR}/a2_${PROFILE}.log"
PID_FILE="${LOG_DIR}/a2_${PROFILE}.pid"

if [[ "${RESUME:-0}" == "1" && -f "${LOG_FILE}" ]]; then
  nohup env RUN_ID="${RUN_ID}" A3_RUN_ID="${A3_RUN_ID:-}" RESUME=1 \
    bash "${SCRIPT_DIR}/run_a2_v2.sh" "${PROFILE}" >>"${LOG_FILE}" 2>&1 </dev/null &
else
  nohup env RUN_ID="${RUN_ID}" A3_RUN_ID="${A3_RUN_ID:-}" RESUME=1 \
    bash "${SCRIPT_DIR}/run_a2_v2.sh" "${PROFILE}" >"${LOG_FILE}" 2>&1 </dev/null &
fi
PID=$!
echo "${PID}" >"${PID_FILE}"
echo "Submitted A2 v2 ${PROFILE}: RUN_ID=${RUN_ID}, PID=${PID}"
echo "tail -f ${LOG_FILE}"
