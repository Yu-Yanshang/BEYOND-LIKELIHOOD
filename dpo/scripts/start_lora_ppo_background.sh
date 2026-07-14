#!/usr/bin/env bash
set -euo pipefail

cd /root/project

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/root/project/dpo/outputs/qwen3_06b_code_lora_ppo_reward_fn/logs}"
PID_FILE="${PID_FILE:-/root/project/dpo/outputs/qwen3_06b_code_lora_ppo_reward_fn/ppo_train.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_${RUN_ID}.log}"

mkdir -p "${LOG_DIR}"

export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
export GPU_ID="${GPU_ID:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

nohup bash -lc 'PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" bash dpo/scripts/train_ppo.sh' >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_FILE}"

echo "pid=${PID}"
echo "log=${LOG_FILE}"
echo "pid_file=${PID_FILE}"
