#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-3}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" dpo/scripts/test_code_dpo.py \
  --gpu_id "${GPU_ID}" \
  --base_config "${BASE_CONFIG:-dpo/configs/qwen3_06b_code_predict_base.yaml}" \
  --dpo_config "${DPO_CONFIG:-dpo/configs/qwen3_06b_code_predict_lora_dpo.yaml}" \
  --base_predictions "${BASE_PREDICTIONS:-dpo/outputs/predict_qwen3_06b_base_on_code_dpo/generated_predictions.jsonl}" \
  --dpo_predictions "${DPO_PREDICTIONS:-dpo/outputs/predict_qwen3_06b_lora_dpo_on_code_dpo/generated_predictions.jsonl}" \
  --output_dir "${OUTPUT_DIR:-dpo/outputs/code_test_qwen3_06b_lora_dpo}" \
  "$@"
