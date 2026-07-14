#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-0}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-./Qwen3-0.6B}
ADAPTER_PATH=${ADAPTER_PATH:-./dpo/outputs/qwen3_06b_code_lora_ppo_reward_fn}
OUTPUT_DIR=${OUTPUT_DIR:-./dpo/outputs/mbpp_eval_ppo_final}
LIMIT=${LIMIT:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
PROMPT_MODE=${PROMPT_MODE:-zero_shot}
CONFIG=${MBPP_CONFIG:-sanitized}
SPLIT=${SPLIT:-test}

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

args=(
  --output_dir "${OUTPUT_DIR}"
  --config "${CONFIG}"
  --split "${SPLIT}"
  --prompt_mode "${PROMPT_MODE}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)

if [[ "${LIMIT}" != "0" ]]; then
  args+=(--limit "${LIMIT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
"${PYTHON_BIN}" dpo/scripts/mbpp_eval_lora_dpo.py \
  --model_path "${BASE_MODEL_PATH}" \
  --adapter_path "${ADAPTER_PATH}" \
  "${args[@]}" >"${OUTPUT_DIR}/run.log" 2>&1

echo "PPO MBPP metrics: ${OUTPUT_DIR}/mbpp_metrics.json"
cat "${OUTPUT_DIR}/mbpp_metrics.json"
