#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/project
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}
RUN_ID=${RUN_ID:-best_path_$(date +%Y%m%d_%H%M%S)}
A3_RUN_ID=${A3_RUN_ID:-a3_v2_verify_20260702_005837}
A2_BATCH_SIZE=${A2_BATCH_SIZE:-12}
A4_BATCH_SIZE=${A4_BATCH_SIZE:-6}
RUN_DIR="${PROJECT_ROOT}/best_path/runs/${RUN_ID}"
COMMAND=${1:-all}

export RUN_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PROJECT_ROOT}/LlamaFactory/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export BNB_CUDA_VERSION=130
export LD_LIBRARY_PATH="/opt/conda/envs/shixun/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"

stage() {
  local name=$1; shift
  local marker="${RUN_DIR}/stages/${name}.done"
  local log="${RUN_DIR}/logs/${name}.log"
  if [[ -f "${marker}" ]]; then echo "[skip] ${name}"; return; fi
  echo "[start] ${name} $(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 2 >"${RUN_DIR}/telemetry/${name}.csv" 2>/dev/null &
  local monitor=$!
  set +e
  "$@" 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  kill "${monitor}" 2>/dev/null || true; wait "${monitor}" 2>/dev/null || true
  [[ ${status} -eq 0 ]] || { echo "[failed] ${name}; see ${log}" >&2; return ${status}; }
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${marker}"
  echo "[done] ${name}"
}

prepare() {
  if [[ -f "${RUN_DIR}/manifest.json" ]]; then return; fi
  "${PYTHON_BIN}" -m best_path.chain prepare --run-id "${RUN_ID}" --a3-run-id "${A3_RUN_ID}" \
    --a2-batch "${A2_BATCH_SIZE}" --a4-batch "${A4_BATCH_SIZE}"
}

train_a2() {
  stage a2_full env A2_TORCH_TELEMETRY="${RUN_DIR}/telemetry/a2_full_torch.json" \
    "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat train "${RUN_DIR}/configs/a2_full.yaml"
}

eval_a2() {
  stage eval_a2 "${PYTHON_BIN}" -m best_path.evaluate --model-path "${RUN_DIR}/models/a2_full" \
    --output-dir "${RUN_DIR}/eval/a2_full" --strategy greedy --batch-size 8 --max-new-tokens 640
}

train_a4() {
  "${PYTHON_BIN}" -m best_path.chain render-a4 --run-id "${RUN_ID}" --a4-batch "${A4_BATCH_SIZE}"
  stage a4_lora_dpo "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat train "${RUN_DIR}/configs/a4_lora_dpo.yaml"
}

merge_a4() {
  "${PYTHON_BIN}" -m best_path.chain render-export --run-id "${RUN_ID}"
  stage merge_a4 "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat export "${RUN_DIR}/configs/a4_export.yaml"
}

eval_a4() {
  stage eval_a4 "${PYTHON_BIN}" -m best_path.evaluate --model-path "${RUN_DIR}/models/a4_dpo_merged" \
    --output-dir "${RUN_DIR}/eval/a4_dpo" --strategy greedy --batch-size 8 --max-new-tokens 640
}

run_a5() {
  stage a5_enhanced "${PYTHON_BIN}" -m best_path.evaluate --model-path "${RUN_DIR}/models/a4_dpo_merged" \
    --output-dir "${RUN_DIR}/eval/a5_enhanced" --strategy enhanced --batch-size 4 \
    --candidates 8 --repairs 2 --max-new-tokens 640 --temperature 0.7 --top-p 0.9
}

make_report() {
  "${PYTHON_BIN}" -m best_path.chain report --run-id "${RUN_ID}" 2>&1 | tee "${RUN_DIR}/logs/report.log"
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${RUN_DIR}/stages/report.done"
}

prepare
case "${COMMAND}" in
  a2) train_a2; eval_a2 ;;
  a4) train_a4; merge_a4; eval_a4 ;;
  a5) run_a5; make_report ;;
  report) make_report ;;
  all) train_a2; eval_a2; train_a4; merge_a4; eval_a4; run_a5; make_report ;;
  *) echo "Usage: RUN_ID=<id> $0 {a2|a4|a5|report|all}" >&2; exit 2 ;;
esac

echo "Best-path command completed: ${COMMAND} (${RUN_ID})"
echo "Report: ${RUN_DIR}/reports/final_report.md"
