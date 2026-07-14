#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/sft/a2_v2/common_env.sh"

COMMAND="${1:-}"
shift || true
if [[ -z "${COMMAND}" ]]; then
  echo "Usage: RUN_ID=<id> $0 {acceptance|formal|prepare|preflight|train|select|export|predict|evaluate|bridge|report} [variant]" >&2
  exit 2
fi

if [[ -z "${RUN_ID:-}" ]]; then
  RUN_ID="a2_v2_$(date +%Y%m%d_%H%M%S)"
  export RUN_ID
fi
DATA_DIR="${PROJECT_ROOT}/sft/data/a2_v2_runs/${RUN_ID}"
OUTPUT_DIR="${PROJECT_ROOT}/sft/outputs/a2_v2_runs/${RUN_ID}"
PIPELINE=("${PYTHON_BIN}" -m sft.a2_v2.pipeline)
PROFILE="${PROFILE:-acceptance}"

prepare_run() {
  local profile="$1"
  local args=(prepare --run-id "${RUN_ID}" --profile "${profile}")
  if [[ "${RESUME:-0}" == "1" ]]; then args+=(--resume); fi
  "${PIPELINE[@]}" "${args[@]}"
}

require_run() {
  [[ -f "${DATA_DIR}/run_manifest.json" ]] || { echo "Missing run; execute prepare first: ${RUN_ID}" >&2; exit 1; }
  PROFILE="$(${PYTHON_BIN} -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "${DATA_DIR}/run_manifest.json")"
  export PROFILE
}

run_stage() {
  local stage="$1"; shift
  local marker="${OUTPUT_DIR}/stages/${stage}.done"
  local log="${OUTPUT_DIR}/logs/${stage}.log"
  local telemetry="${OUTPUT_DIR}/telemetry/${stage}.csv"
  if [[ -f "${marker}" && "${RESUME:-0}" == "1" ]]; then
    echo "[skip] ${stage} already completed"
    return 0
  fi
  echo "[start] ${stage} $(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -l 1 >"${telemetry}" 2>/dev/null &
  local monitor_pid=$!
  set +e
  "$@" 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  if [[ ${status} -ne 0 ]]; then
    echo "[failed] ${stage}; see ${log}" >&2
    return "${status}"
  fi
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"${marker}"
  echo "[done] ${stage}"
}

do_preflight() {
  run_stage preflight "${PYTHON_BIN}" -m sft.a2_v2.preflight \
    --model-path "${PROJECT_ROOT}/Qwen3-0.6B" --output "${OUTPUT_DIR}/reports/preflight.json"
}

train_one() {
  local variant="$1"
  [[ " ${VARIANTS[*]} " == *" ${variant} "* ]] || { echo "Unknown variant: ${variant}" >&2; return 2; }
  run_stage "train_${variant}" env A2_TORCH_TELEMETRY="${OUTPUT_DIR}/telemetry/train_${variant}_torch.json" \
    "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat train "${OUTPUT_DIR}/configs/train_${variant}.yaml"
}

do_train() {
  local variant="${1:-all}"
  local VARIANTS=(full qlora qdora)
  if [[ "${variant}" == "all" ]]; then
    for item in "${VARIANTS[@]}"; do train_one "${item}"; done
  else
    train_one "${variant}"
  fi
}

do_validate_artifacts() {
  run_stage validate_artifacts "${PIPELINE[@]}" validate-artifacts --run-id "${RUN_ID}"
}

selection_body() {
  if [[ "${PROFILE}" == "formal" ]]; then
    for variant in full qlora qdora; do
      shopt -s nullglob
      checkpoints=("${OUTPUT_DIR}/models/${variant}"/checkpoint-*)
      shopt -u nullglob
      if [[ ${#checkpoints[@]} -eq 0 ]]; then checkpoints=("${OUTPUT_DIR}/models/${variant}"); fi
      for checkpoint in "${checkpoints[@]}"; do
        name="$(basename "${checkpoint}")"
        tag="selection_${name}"
        config="$(${PIPELINE[@]} render-predict --run-id "${RUN_ID}" --variant "${variant}" \
          --dataset a2_mbpp_select --source-path "${checkpoint}" --tag "${tag}")"
        "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat train "${config}"
        pred_dir="${OUTPUT_DIR}/predictions/${tag}/${variant}"
        eval_dir="${OUTPUT_DIR}/eval/selection/${variant}/${name}"
        "${PYTHON_BIN}" "${PROJECT_ROOT}/sft/scripts/evaluate_code_predictions.py" \
          --predictions "${pred_dir}/generated_predictions.jsonl" --mbpp_dir "${PROJECT_ROOT}/../mbpp" \
          --config sanitized --split test --output_dir "${eval_dir}" --limit 64
      done
    done
  fi
  "${PIPELINE[@]}" select --run-id "${RUN_ID}"
}

do_select() { run_stage select selection_body; }

selected_field() {
  local variant="$1" field="$2"
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/selected_models.json" "${variant}" "${field}" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))["variants"][sys.argv[2]].get(sys.argv[3])
print(value or "")
PY
}

export_body() {
  for variant in qlora qdora; do
    adapter="$(selected_field "${variant}" adapter_path)"
    config="$(${PIPELINE[@]} render-export --run-id "${RUN_ID}" --variant "${variant}" --adapter-path "${adapter}")"
    "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat export "${config}"
  done
}

do_export() { run_stage export export_body; }

predict_body() {
  local variant dataset tag source extra=()
  for variant in base full qlora qdora; do
    source=""
    extra=()
    if [[ "${variant}" == "full" ]]; then
      source="$(selected_field full model_path)"
      extra=(--source-path "${source}" --full-model)
    elif [[ "${variant}" == "qlora" || "${variant}" == "qdora" ]]; then
      source="${OUTPUT_DIR}/merged/${variant}"
      extra=(--source-path "${source}" --full-model)
    fi
    for dataset in a2_test a2_mbpp; do
      tag="final_${dataset#a2_}"
      config="$(${PIPELINE[@]} render-predict --run-id "${RUN_ID}" --variant "${variant}" \
        --dataset "${dataset}" --tag "${tag}" "${extra[@]}")"
      "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat train "${config}"
    done
  done
}

do_predict() { run_stage predict predict_body; }

evaluate_body() {
  local limit=0
  [[ "${PROFILE}" == "acceptance" ]] && limit=32
  for variant in base full qlora qdora; do
    "${PYTHON_BIN}" "${PROJECT_ROOT}/sft/scripts/evaluate_code_predictions.py" \
      --predictions "${OUTPUT_DIR}/predictions/final_mbpp/${variant}/generated_predictions.jsonl" \
      --mbpp_dir "${PROJECT_ROOT}/../mbpp" --config sanitized --split test \
      --output_dir "${OUTPUT_DIR}/eval/final/${variant}" --limit "${limit}"
    "${PIPELINE[@]}" evaluate-a1 \
      --predictions "${OUTPUT_DIR}/predictions/final_test/${variant}/generated_predictions.jsonl" \
      --references "${DATA_DIR}/a2_test.json" --output-dir "${OUTPUT_DIR}/eval/final/${variant}"
  done
}

do_evaluate() { run_stage evaluate evaluate_body; }

do_report() { run_stage report "${PIPELINE[@]}" report --run-id "${RUN_ID}"; }

do_bridge() {
  run_stage bridge env RUN_ID="${RUN_ID}" A3_RUN_ID="${A3_RUN_ID:-}" \
    bash "${PROJECT_ROOT}/sft/scripts/validate_a2_a3_a4_bridge.sh"
}

workflow() {
  local profile="$1"
  PROFILE="${profile}"; export PROFILE
  prepare_run "${profile}"
  require_run
  do_preflight
  do_train all
  do_validate_artifacts
  do_select
  do_export
  do_predict
  do_evaluate
  do_report
  do_bridge
  echo "A2 v2 ${profile} completed: ${RUN_ID}"
  echo "Report: ${OUTPUT_DIR}/reports/comparison_report.md"
  echo "Handoff: ${OUTPUT_DIR}/a2_handoff_manifest.json"
}

case "${COMMAND}" in
  acceptance) workflow acceptance ;;
  formal|all) workflow formal ;;
  prepare) prepare_run "${PROFILE}" ;;
  preflight) require_run; do_preflight ;;
  train) require_run; VARIANTS=(full qlora qdora); do_train "${1:-all}" ;;
  select) require_run; do_select ;;
  export) require_run; do_export ;;
  predict) require_run; do_predict ;;
  evaluate) require_run; do_evaluate ;;
  report) require_run; do_report ;;
  bridge) require_run; do_bridge ;;
  *) echo "Unknown command: ${COMMAND}" >&2; exit 2 ;;
esac
