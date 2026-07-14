#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" tts/showcase_cli.py \
  --model_path "${MODEL_PATH:-${PROJECT_ROOT}/best_path/runs/best_full_dpo_a5_20260702/models/a2_full}" \
  --output_dir "${OUTPUT_DIR:-${PROJECT_ROOT}/tts/outputs/showcase_sessions}" \
  "$@"
