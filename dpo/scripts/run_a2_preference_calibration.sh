#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"
GPU_ID="${GPU_ID:-0}"

RUN_ID="${RUN_ID:-a2_pref_calib_$(date +%Y%m%d_%H%M%S)}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/best_path/runs/best_full_dpo_a5_20260702/models/a2_full}"
REF_MODEL="${REF_MODEL:-${BASE_MODEL}}"
SOURCE_DATA="${SOURCE_DATA:-${PROJECT_ROOT}/best_path/runs/best_full_dpo_a5_20260702/data/a3_high_confidence.json}"

TRAIN_NAME="${TRAIN_NAME:-a2_pref_calib_train}"
EVAL_NAME="${EVAL_NAME:-a2_pref_calib_eval}"
MAX_TRAIN="${MAX_TRAIN:-2000}"
EVAL_SIZE="${EVAL_SIZE:-256}"
DATA_SEED="${DATA_SEED:-20260708}"

PREF_BETA="${PREF_BETA:-0.05}"
PREF_FTX="${PREF_FTX:-0.20}"
PREF_LOSS="${PREF_LOSS:-sigmoid}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET="${LORA_TARGET:-q_proj,v_proj}"
CUTOFF_LEN="${CUTOFF_LEN:-1536}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
BASELINE_METRICS="${BASELINE_METRICS:-${PROJECT_ROOT}/best_path/runs/best_full_dpo_a5_20260702/eval/a2_full/metrics.json}"
MIN_PASS_AT_1="${MIN_PASS_AT_1:-0.4980544747081712}"
ENFORCE_GATE="${ENFORCE_GATE:-1}"
DRY_RUN="${DRY_RUN:-0}"

ROOT_OUT="${PROJECT_ROOT}/dpo/outputs/${RUN_ID}"
ADAPTER_OUT="${ROOT_OUT}/adapter"
LOG_DIR="${ROOT_OUT}/logs"
RECORD_DIR="${ROOT_OUT}/records"
EVAL_DIR="${ROOT_OUT}/mbpp_eval"
CONFIG_PATH="${RECORD_DIR}/a2_preference_calibration_lora_dpo.yaml"
REPORT_PATH="${ROOT_OUT}/CALIBRATION_REPORT.json"
RECORD_MD="${ROOT_OUT}/TRAINING_RECORD.md"
PID_FILE="${ROOT_OUT}/run.pid"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${PROJECT_ROOT}:${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${ADAPTER_OUT}" "${LOG_DIR}" "${RECORD_DIR}" "${EVAL_DIR}"
echo "$$" >"${PID_FILE}"

record() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*" | tee -a "${RECORD_MD}"
}

latest_checkpoint() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
  fi
}

render_config() {
  cat >"${CONFIG_PATH}" <<EOF
### model
model_name_or_path: ${BASE_MODEL}
trust_remote_code: true
ref_model: ${REF_MODEL}

### method
stage: dpo
do_train: true
do_eval: true
finetuning_type: lora
pref_beta: ${PREF_BETA}
pref_ftx: ${PREF_FTX}
pref_loss: ${PREF_LOSS}

### LoRA
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}
lora_target: ${LORA_TARGET}

### dataset
dataset_dir: ./dpo/data
dataset: ${TRAIN_NAME}
eval_dataset: ${EVAL_NAME}
template: qwen3
enable_thinking: false
cutoff_len: ${CUTOFF_LEN}
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 2

### output
output_dir: ${ADAPTER_OUT}
logging_steps: 5
eval_strategy: steps
eval_steps: 25
save_strategy: steps
save_steps: 25
save_total_limit: 2
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
plot_loss: true
overwrite_output_dir: false
save_only_model: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
learning_rate: ${LEARNING_RATE}
num_train_epochs: ${NUM_TRAIN_EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: ${WARMUP_RATIO}
max_grad_norm: 0.5
gradient_checkpointing: true
bf16: true
fp16: false
ddp_timeout: 180000000
resume_from_checkpoint: null
EOF
}

prepare_data() {
  record "Prepare conservative preference subset: source=${SOURCE_DATA}, train=${MAX_TRAIN}, eval=${EVAL_SIZE}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/prepare_a2_preference_calibration_data.py" \
    --source "${SOURCE_DATA}" \
    --output-dir "${PROJECT_ROOT}/dpo/data" \
    --train-name "${TRAIN_NAME}" \
    --eval-name "${EVAL_NAME}" \
    --max-train "${MAX_TRAIN}" \
    --eval-size "${EVAL_SIZE}" \
    --seed "${DATA_SEED}" | tee "${LOG_DIR}/prepare_data.log"
}

run_train() {
  if [[ -f "${ADAPTER_OUT}/adapter_model.safetensors" || -f "${ADAPTER_OUT}/all_results.json" ]]; then
    record "SKIP train: existing adapter or final results found in ${ADAPTER_OUT}."
    return
  fi

  render_config
  record "Rendered config: ${CONFIG_PATH}."
  record "START conservative LoRA-DPO calibration from A2 Full SFT."
  CONFIG="${CONFIG_PATH}" PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" \
    bash "${PROJECT_ROOT}/dpo/scripts/train_with_compat.sh" 2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
  record "END train: adapter=${ADAPTER_OUT}."
}

adapter_for_eval() {
  if [[ -f "${ADAPTER_OUT}/adapter_model.safetensors" || -f "${ADAPTER_OUT}/adapter_config.json" ]]; then
    printf '%s\n' "${ADAPTER_OUT}"
    return
  fi
  latest_checkpoint "${ADAPTER_OUT}"
}

run_eval() {
  local eval_adapter
  eval_adapter="$(adapter_for_eval)"
  if [[ -z "${eval_adapter}" ]]; then
    record "ERROR: no adapter found under ${ADAPTER_OUT}."
    return 1
  fi

  local outdir="${EVAL_DIR}/a2_pref_calib"
  if [[ -f "${outdir}/mbpp_metrics.json" ]]; then
    record "SKIP eval: ${outdir}/mbpp_metrics.json exists."
  else
    record "START MBPP eval: base=${BASE_MODEL}, adapter=${eval_adapter}."
    BASE_MODEL_PATH="${BASE_MODEL}" ADAPTER_PATH="${eval_adapter}" OUTPUT_DIR="${outdir}" \
      PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" BATCH_SIZE="${EVAL_BATCH_SIZE}" MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}" \
      bash "${PROJECT_ROOT}/dpo/scripts/run_mbpp_lora_dpo_eval.sh" 2>&1 | tee "${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
    record "END MBPP eval: ${outdir}."
  fi
}

write_report_and_gate() {
  local metrics="${EVAL_DIR}/a2_pref_calib/mbpp_metrics.json"
  "${PYTHON_BIN}" - "${BASELINE_METRICS}" "${metrics}" "${REPORT_PATH}" "${MIN_PASS_AT_1}" <<'PY'
import json
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
floor = float(sys.argv[4])

baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

baseline_pass = baseline.get("pass_at_1", floor)
pass_at_1 = metrics.get("pass_at_1")
delta = None if pass_at_1 is None or baseline_pass is None else pass_at_1 - baseline_pass
gate_passed = pass_at_1 is not None and pass_at_1 >= floor

report = {
    "baseline_metrics": str(baseline_path),
    "eval_metrics": str(metrics_path),
    "baseline_pass_at_1": baseline_pass,
    "min_pass_at_1": floor,
    "calibrated_pass_at_1": pass_at_1,
    "delta_vs_baseline": delta,
    "gate_passed": gate_passed,
    "calibrated_metrics": metrics,
}
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
sys.exit(0 if gate_passed else 2)
PY
}

main() {
  cd "${PROJECT_ROOT}"
  if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "Missing BASE_MODEL: ${BASE_MODEL}" >&2
    exit 1
  fi

  record "RUN START: run_id=${RUN_ID}, gpu=${GPU_ID}, python=${PYTHON_BIN}."
  record "Policy: LoRA-only small calibration, A2 as ref_model, pref_ftx=${PREF_FTX}, lr=${LEARNING_RATE}, epochs=${NUM_TRAIN_EPOCHS}."
  prepare_data
  if [[ "${DRY_RUN}" == "1" ]]; then
    render_config
    record "DRY RUN COMPLETE: rendered config=${CONFIG_PATH}; no training or evaluation started."
    return
  fi
  run_train
  run_eval

  if write_report_and_gate; then
    record "GATE PASS: calibrated pass_at_1 >= ${MIN_PASS_AT_1}. Report=${REPORT_PATH}."
  else
    local status=$?
    record "GATE FAIL: calibrated pass_at_1 < ${MIN_PASS_AT_1}. Keep A2 Full SFT as production model. Report=${REPORT_PATH}."
    if [[ "${ENFORCE_GATE}" == "1" ]]; then
      exit "${status}"
    fi
  fi

  record "RUN COMPLETE."
}

main "$@"
