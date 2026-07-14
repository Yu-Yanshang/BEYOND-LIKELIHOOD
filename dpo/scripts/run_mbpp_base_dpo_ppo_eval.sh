#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
RUN_PARALLEL=${RUN_PARALLEL:-0}

BASE_GPUS=${BASE_GPUS:-0}
DPO_GPUS=${DPO_GPUS:-0}
PPO_GPUS=${PPO_GPUS:-0}

BASE_MODEL_PATH=${BASE_MODEL_PATH:-./Qwen3-0.6B}
DPO_MODEL_PATH=${DPO_MODEL_PATH:-./dpo/outputs/qwen3_06b_code_full_dpo}
PPO_MODEL_PATH=${PPO_MODEL_PATH:-./dpo/outputs/qwen3_06b_code_full_ppo}

BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-./dpo/outputs/mbpp_eval_base}
DPO_OUTPUT_DIR=${DPO_OUTPUT_DIR:-./dpo/outputs/mbpp_eval_dpo_final}
PPO_OUTPUT_DIR=${PPO_OUTPUT_DIR:-./dpo/outputs/mbpp_eval_ppo_final}
COMPARE_OUTPUT=${COMPARE_OUTPUT:-./dpo/outputs/mbpp_base_dpo_ppo_compare/mbpp_base_vs_dpo_vs_ppo_metrics.json}

LIMIT=${LIMIT:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
PROMPT_MODE=${PROMPT_MODE:-zero_shot}
CONFIG=${MBPP_CONFIG:-sanitized}
SPLIT=${SPLIT:-test}

cd "${PROJECT_ROOT}"

common_args=(
  --config "${CONFIG}"
  --split "${SPLIT}"
  --prompt_mode "${PROMPT_MODE}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
)
if [[ "${LIMIT}" != "0" ]]; then
  common_args+=(--limit "${LIMIT}")
fi

run_one() {
  local name="$1"
  local gpus="$2"
  local model_path="$3"
  local output_dir="$4"
  mkdir -p "${output_dir}"
  echo "[${name}] GPUs=${gpus} model=${model_path}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON_BIN}" dpo/scripts/mbpp_eval_dpo.py \
    --model_path "${model_path}" \
    --output_dir "${output_dir}" \
    "${common_args[@]}" \
    >"${output_dir}/run.log" 2>&1
}

if [[ "${RUN_PARALLEL}" == "1" ]]; then
  run_one base "${BASE_GPUS}" "${BASE_MODEL_PATH}" "${BASE_OUTPUT_DIR}" & base_pid=$!
  run_one dpo "${DPO_GPUS}" "${DPO_MODEL_PATH}" "${DPO_OUTPUT_DIR}" & dpo_pid=$!
  run_one ppo "${PPO_GPUS}" "${PPO_MODEL_PATH}" "${PPO_OUTPUT_DIR}" & ppo_pid=$!
  wait "${base_pid}"
  wait "${dpo_pid}"
  wait "${ppo_pid}"
else
  run_one base "${BASE_GPUS}" "${BASE_MODEL_PATH}" "${BASE_OUTPUT_DIR}"
  run_one dpo "${DPO_GPUS}" "${DPO_MODEL_PATH}" "${DPO_OUTPUT_DIR}"
  run_one ppo "${PPO_GPUS}" "${PPO_MODEL_PATH}" "${PPO_OUTPUT_DIR}"
fi

"${PYTHON_BIN}" dpo/scripts/compare_mbpp_base_dpo_ppo_metrics.py \
  --base_metrics "${BASE_OUTPUT_DIR}/mbpp_metrics.json" \
  --dpo_metrics "${DPO_OUTPUT_DIR}/mbpp_metrics.json" \
  --ppo_metrics "${PPO_OUTPUT_DIR}/mbpp_metrics.json" \
  --output "${COMPARE_OUTPUT}"
