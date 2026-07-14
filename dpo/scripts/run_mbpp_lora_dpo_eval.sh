#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"

DEFAULT_PYTHON="/data/yekaiyang/anaconda3/envs/assignment_A/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" dpo/scripts/mbpp_eval_lora_dpo.py \
  --config "${MBPP_CONFIG:-sanitized}" \
  --split "${MBPP_SPLIT:-test}" \
  --prompt_mode "${PROMPT_MODE:-zero_shot}" \
  --model_path "${BASE_MODEL_PATH:-./Qwen3-0.6B}" \
  --adapter_path "${ADAPTER_PATH:-dpo/outputs/qwen3_06b_code_lora_dpo}" \
  --output_dir "${OUTPUT_DIR:-dpo/outputs/mbpp_eval_qwen3_06b_lora_dpo}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  "$@"
