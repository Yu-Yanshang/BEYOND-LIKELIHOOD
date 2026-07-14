#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/project
PYTHON_BIN=/opt/conda/envs/shixun/bin/python
RUN_ID=${RUN_ID:-best_full_dpo_a5_20260702}
RUN_DIR="${PROJECT_ROOT}/best_path/runs/${RUN_ID}"
export PYTHONPATH="${PROJECT_ROOT}/LlamaFactory/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "${PROJECT_ROOT}"

wait_for_full_a5() {
  while pgrep -f "best_path.evaluate --model-path .*${RUN_ID}/models/a2_full .*${RUN_ID}/eval/a5_enhanced" >/dev/null; do
    echo "[$(date --iso-8601=seconds)] waiting for Full-SFT -> A5"
    sleep 60
  done
  if [[ ! -f "${RUN_DIR}/eval/a5_enhanced/metrics.json" ]]; then
    echo "[$(date --iso-8601=seconds)] prior Full-SFT -> A5 did not finish; restarting it"
    "${PYTHON_BIN}" -m best_path.evaluate \
      --model-path "${RUN_DIR}/models/a2_full" --output-dir "${RUN_DIR}/eval/a5_enhanced" \
      --strategy enhanced --batch-size 4 --candidates 8 --repairs 2 --max-new-tokens 640 \
      --temperature 0.7 --top-p 0.9 2>&1 | tee -a "${RUN_DIR}/logs/a5_enhanced.log"
  fi
  if [[ ! -e "${RUN_DIR}/eval/a5_from_a2_full" ]]; then
    mv "${RUN_DIR}/eval/a5_enhanced" "${RUN_DIR}/eval/a5_from_a2_full"
  fi
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${RUN_DIR}/stages/a5_from_a2_full.done"
}

legacy_a4_eval() {
  if [[ -f "${RUN_DIR}/eval/a4_dpo_legacy_protocol/mbpp_metrics.json" ]]; then return; fi
  "${PYTHON_BIN}" dpo/scripts/mbpp_eval_dpo.py \
    --config sanitized --split test --prompt_mode zero_shot \
    --model_path "${RUN_DIR}/models/a4_dpo_merged" \
    --output_dir "${RUN_DIR}/eval/a4_dpo_legacy_protocol" \
    --batch_size 8 --max_new_tokens 512 \
    2>&1 | tee "${RUN_DIR}/logs/eval_a4_legacy_protocol.log"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${RUN_DIR}/stages/eval_a4_legacy_protocol.done"
}

dpo_a5() {
  if [[ -f "${RUN_DIR}/eval/a5_from_a4_dpo/metrics.json" ]]; then return; fi
  "${PYTHON_BIN}" -m best_path.evaluate \
    --model-path "${RUN_DIR}/models/a4_dpo_merged" --output-dir "${RUN_DIR}/eval/a5_from_a4_dpo" \
    --strategy enhanced --batch-size 4 --candidates 8 --repairs 2 --max-new-tokens 640 \
    --temperature 0.7 --top-p 0.9 2>&1 | tee "${RUN_DIR}/logs/a5_from_a4_dpo.log"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${RUN_DIR}/stages/a5_from_a4_dpo.done"
}

finalize() {
  "${PYTHON_BIN}" -m best_path.finalize --run-dir "${RUN_DIR}" 2>&1 | tee "${RUN_DIR}/logs/finalize.log"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${RUN_DIR}/stages/dual_route_report.done"
}

echo "[$(date --iso-8601=seconds)] detached continuation started for ${RUN_ID}"
wait_for_full_a5
legacy_a4_eval
dpo_a5
finalize
echo "[$(date --iso-8601=seconds)] ALL_DONE report=${RUN_DIR}/reports/final_report.md"
