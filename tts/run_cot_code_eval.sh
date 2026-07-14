#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="/opt/conda/envs/shixun/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" tts/evaluate_cot_code.py \
  --model_path "${MODEL_PATH:-Qwen3-0.6B}" \
  --input_file "${INPUT_FILE:-sft/data/mbpp_sanitized_test.json}" \
  --output_dir "${OUTPUT_DIR:-tts/outputs/cot_code_eval}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-768}" \
  "$@"
