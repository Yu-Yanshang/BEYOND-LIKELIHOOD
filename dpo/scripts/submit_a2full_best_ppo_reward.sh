#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-a2full_best_ppo_reward_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${PROJECT_ROOT}/dpo/outputs/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/submit.log"
PID_FILE="${RUN_ROOT}/submit.pid"

cd "${PROJECT_ROOT}"
RUN_ID="${RUN_ID}" nohup bash dpo/scripts/run_a2full_best_ppo_reward.sh >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"

echo "Submitted A2-full PPO reward run."
echo "RUN_ID=${RUN_ID}"
echo "PID=${PID}"
echo "Log=${LOG_FILE}"
echo "Record=${RUN_ROOT}/TRAINING_RECORD.md"
