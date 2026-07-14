#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/root/project
RUN_ID=${RUN_ID:-best_path_$(date +%Y%m%d_%H%M%S)}
COMMAND=${1:-all}
SUBMIT_DIR="${PROJECT_ROOT}/best_path/submissions"
mkdir -p "${SUBMIT_DIR}"
LOG="${SUBMIT_DIR}/${RUN_ID}.log"
PID_FILE="${SUBMIT_DIR}/${RUN_ID}.pid"
RUN_ID="${RUN_ID}" nohup bash "${PROJECT_ROOT}/best_path/scripts/run_full_chain.sh" "${COMMAND}" >"${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" >"${PID_FILE}"
echo "Submitted PID ${pid}"
echo "Bootstrap log: ${LOG}"
echo "Follow: tail -f ${LOG}"
echo "After prepare: tail -f ${PROJECT_ROOT}/best_path/runs/${RUN_ID}/logs/*.log"
