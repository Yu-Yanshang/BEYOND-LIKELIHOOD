#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ID="${RUN_ID:-a2full_dpo_lora_ppo_20260706}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"

ROOT_OUT="${PROJECT_ROOT}/dpo/outputs/${RUN_ID}"
LOG_DIR="${ROOT_OUT}/logs"
RECORD_DIR="${ROOT_OUT}/records"
CONFIG_DIR="${RECORD_DIR}/rendered_configs"
EVAL_DIR="${ROOT_OUT}/mbpp_eval"
RECORD_MD="${ROOT_OUT}/TRAINING_RECORD.md"
PID_FILE="${ROOT_OUT}/sequence.pid"

FULL_OUT="${ROOT_OUT}/full_dpo"
LORA_OUT="${ROOT_OUT}/lora_dpo"
PPO_OUT="${ROOT_OUT}/ppo_reward_fn"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-18080}"
REWARD_SERVER_PID=""

FULL_CONFIG="${PROJECT_ROOT}/dpo/configs/a2full_chain_full_dpo.yaml"
LORA_CONFIG="${PROJECT_ROOT}/dpo/configs/a2full_chain_lora_dpo.yaml"
PPO_CONFIG="${PROJECT_ROOT}/dpo/configs/a2full_chain_ppo_reward_fn.yaml"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${LOG_DIR}" "${RECORD_DIR}" "${CONFIG_DIR}" "${EVAL_DIR}"
echo "$$" >"${PID_FILE}"

record() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*" | tee -a "${RECORD_MD}"
}

stop_reward_server() {
  if [[ -n "${REWARD_SERVER_PID}" ]]; then
    kill "${REWARD_SERVER_PID}" 2>/dev/null || true
    wait "${REWARD_SERVER_PID}" 2>/dev/null || true
  fi
}

start_reward_server() {
  local log="${LOG_DIR}/reward_server_$(date +%Y%m%d_%H%M%S).log"
  record "START reward server: http://127.0.0.1:${REWARD_SERVER_PORT}, log=${log}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/code_reward_server.py" \
    --host 127.0.0.1 \
    --port "${REWARD_SERVER_PORT}" >"${log}" 2>&1 &
  REWARD_SERVER_PID=$!
  trap stop_reward_server EXIT
  sleep 3
}

init_record() {
  if [[ ! -f "${RECORD_MD}" ]]; then
    cat >"${RECORD_MD}" <<EOF
# A2 Full Base DPO/LoRA/PPO Training Record

- Run ID: ${RUN_ID}
- Project root: ${PROJECT_ROOT}
- Conda env: /opt/conda/envs/shixun
- Base model: /root/project/best_path/runs/best_full_dpo_a5_20260702/models/a2_full
- Ordered stages: Full DPO -> LoRA DPO -> PPO with local reward function
- Epoch limit: each train stage is configured with num_train_epochs <= 3.0
- Early stopping:
  - Full DPO and LoRA DPO monitor eval_loss with load_best_model_at_end.
  - PPO monitors logged reward and stops after configured non-improving logging events.
- Resume policy:
  - Full DPO and LoRA DPO resume from the newest checkpoint-* directory.
  - PPO resumes by loading the newest PPO adapter checkpoint as adapter_name_or_path.
- Preservation policy: model files, trainer state, checkpoints, loss curves, logs, and MBPP eval outputs remain under ${ROOT_OUT}.

## Timeline
EOF
  fi
}

latest_checkpoint() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
  fi
}

latest_logged_epoch() {
  local outdir="$1"
  local log="${outdir}/trainer_log.jsonl"
  if [[ -f "${log}" ]]; then
    "${PYTHON_BIN}" - "${log}" <<'PY'
import json
import sys
from pathlib import Path

latest = 0.0
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    epoch = row.get("epoch")
    if isinstance(epoch, (int, float)):
        latest = max(latest, float(epoch))
print(f"{latest:.6f}")
PY
  else
    printf '0\n'
  fi
}

weight_resume_dir() {
  local stage="$1"
  local checkpoint="$2"
  local resume_root="${ROOT_OUT}/resume_models"
  local alias="${resume_root}/${stage}_from_$(basename "${checkpoint}")"
  mkdir -p "${alias}"
  find "${alias}" -mindepth 1 -maxdepth 1 -type l -delete
  find "${checkpoint}" -maxdepth 1 -type f \
    ! -name 'optimizer.pt' \
    ! -name 'scheduler.pt' \
    ! -name 'rng_state*.pth' \
    -exec ln -sfn {} "${alias}/" \;
  printf '%s\n' "${alias}"
}

render_config() {
  local template="$1"
  local output="$2"
  local mode="$3"
  local checkpoint="${4:-}"
  local adapter="${5:-}"
  local epochs_already="${6:-0}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/render_resume_config.py" \
    --template "${template}" \
    --output "${output}" \
    --resume-mode "${mode}" \
    --resume-checkpoint "${checkpoint}" \
    --adapter-path "${adapter}" \
    --epochs-already "${epochs_already}"
}

run_train_stage() {
  local stage="$1"
  local template="$2"
  local outdir="$3"
  local runner="$4"
  local resume_mode="$5"
  local base_adapter="${6:-}"

  if [[ -f "${outdir}/train_results.json" || -f "${outdir}/all_results.json" ]]; then
    record "SKIP ${stage}: existing final train result found in ${outdir}."
    return
  fi

  local ckpt
  ckpt="$(latest_checkpoint "${outdir}" || true)"
  local epochs_already="0"
  local render_checkpoint="${ckpt}"
  if [[ -n "${ckpt}" && "${resume_mode}" != "trainer" ]]; then
    epochs_already="$(latest_logged_epoch "${outdir}")"
    if [[ "${resume_mode}" == "model" ]]; then
      render_checkpoint="$(weight_resume_dir "${stage}" "${ckpt}")"
    fi
  fi
  local rendered="${CONFIG_DIR}/${stage}.yaml"
  render_config "${template}" "${rendered}" "${resume_mode}" "${render_checkpoint}" "${base_adapter}" "${epochs_already}"

  if [[ -n "${ckpt}" ]]; then
    record "START ${stage}: weight-level resume from ${ckpt}; epochs_already=${epochs_already}; rendered_checkpoint=${render_checkpoint}."
  else
    record "START ${stage}: starting from configured base."
  fi

  local log="${LOG_DIR}/${stage}_$(date +%Y%m%d_%H%M%S).log"
  CONFIG="${rendered}" PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" bash "${runner}" 2>&1 | tee "${log}"
  record "END ${stage}: log=${log}."
}

run_eval_full() {
  local stage="$1"
  local model_path="$2"
  local outdir="${EVAL_DIR}/${stage}"
  if [[ -f "${outdir}/mbpp_metrics.json" ]]; then
    record "SKIP eval ${stage}: ${outdir}/mbpp_metrics.json exists."
    return
  fi
  record "START eval ${stage}."
  DPO_MODEL_PATH="${model_path}" OUTPUT_DIR="${outdir}" PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" \
    bash "${PROJECT_ROOT}/dpo/scripts/run_mbpp_dpo_final_eval.sh" 2>&1 | tee "${LOG_DIR}/eval_${stage}_$(date +%Y%m%d_%H%M%S).log"
  record "END eval ${stage}: ${outdir}."
}

run_eval_lora() {
  local stage="$1"
  local base_model="$2"
  local adapter="$3"
  local outdir="${EVAL_DIR}/${stage}"
  if [[ -f "${outdir}/mbpp_metrics.json" ]]; then
    record "SKIP eval ${stage}: ${outdir}/mbpp_metrics.json exists."
    return
  fi
  record "START eval ${stage}."
  BASE_MODEL_PATH="${base_model}" ADAPTER_PATH="${adapter}" OUTPUT_DIR="${outdir}" PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" \
    bash "${PROJECT_ROOT}/dpo/scripts/run_mbpp_lora_dpo_eval.sh" 2>&1 | tee "${LOG_DIR}/eval_${stage}_$(date +%Y%m%d_%H%M%S).log"
  record "END eval ${stage}: ${outdir}."
}

main() {
  cd "${PROJECT_ROOT}"
  init_record
  record "RUN START: pid=$$, gpu=${GPU_ID}, python=${PYTHON_BIN}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/prepare_chain_dpo_data.py" | tee -a "${RECORD_MD}"

  run_train_stage "full_dpo" "${FULL_CONFIG}" "${FULL_OUT}" "${PROJECT_ROOT}/dpo/scripts/train_with_compat.sh" "model"
  run_eval_full "full_dpo" "${FULL_OUT}"

  run_train_stage "lora_dpo" "${LORA_CONFIG}" "${LORA_OUT}" "${PROJECT_ROOT}/dpo/scripts/train_with_compat.sh" "adapter"
  run_eval_lora "lora_dpo" "${FULL_OUT}" "${LORA_OUT}"

  local ppo_base_adapter="${LORA_OUT}"
  start_reward_server
  run_train_stage "ppo_reward_fn" "${PPO_CONFIG}" "${PPO_OUT}" "${PROJECT_ROOT}/dpo/scripts/run_ppo_with_reward_api_monitor.sh" "adapter" "${ppo_base_adapter}"
  local ppo_eval_adapter="${PPO_OUT}"
  if [[ ! -f "${ppo_eval_adapter}/adapter_model.safetensors" ]]; then
    ppo_eval_adapter="$(latest_checkpoint "${PPO_OUT}" || true)"
  fi
  run_eval_lora "ppo_reward_fn" "${FULL_OUT}" "${ppo_eval_adapter}"

  record "RUN COMPLETE."
}

main "$@"
