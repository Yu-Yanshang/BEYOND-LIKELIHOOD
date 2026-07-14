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
PID_FILE="${ROOT_OUT}/epoch_continue_sequence.pid"

FULL_OUT="${ROOT_OUT}/full_dpo_epoch_continue"
LORA_OUT="${ROOT_OUT}/lora_dpo_epoch_continue"
PPO_OUT="${ROOT_OUT}/ppo_reward_fn_epoch_continue"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-18080}"
REWARD_SERVER_PID=""

FULL_CONFIG="${PROJECT_ROOT}/dpo/configs/a2full_epoch_continue_full_dpo.yaml"
LORA_CONFIG="${PROJECT_ROOT}/dpo/configs/a2full_epoch_continue_lora_dpo.yaml"
PPO_CONFIG="${PROJECT_ROOT}/dpo/configs/a2full_epoch_continue_ppo_reward_fn.yaml"
PPO_BASE_MODEL="${PPO_BASE_MODEL:-/root/project/best_path/runs/best_full_dpo_a5_20260702/models/a2_full}"

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

latest_checkpoint() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
  fi
}

render_config() {
  local template="$1"
  local output="$2"
  local mode="$3"
  local checkpoint="${4:-}"
  local adapter="${5:-}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/render_resume_config.py" \
    --template "${template}" \
    --output "${output}" \
    --resume-mode "${mode}" \
    --resume-checkpoint "${checkpoint}" \
    --adapter-path "${adapter}"
}

run_train_stage() {
  local stage="$1"
  local template="$2"
  local outdir="$3"
  local runner="$4"
  local resume_mode="${5:-none}"

  if [[ -f "${outdir}/train_results.json" || -f "${outdir}/all_results.json" ]]; then
    record "SKIP ${stage}: existing final train result found in ${outdir}."
    return
  fi

  local ckpt
  ckpt="$(latest_checkpoint "${outdir}" || true)"
  local rendered="${CONFIG_DIR}/${stage}.yaml"
  if [[ -n "${ckpt}" ]]; then
    render_config "${template}" "${rendered}" "${resume_mode}" "${ckpt}"
    record "START ${stage}: resuming from ${ckpt} with epoch-level eval/early-stop."
  else
    render_config "${template}" "${rendered}" "none"
    record "START ${stage}: starting from configured existing base with epoch-level eval/early-stop."
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

stop_reward_server() {
  if [[ -n "${REWARD_SERVER_PID}" ]]; then
    kill "${REWARD_SERVER_PID}" 2>/dev/null || true
    wait "${REWARD_SERVER_PID}" 2>/dev/null || true
  fi
}

start_reward_server() {
  local log="${LOG_DIR}/reward_server_epoch_continue_$(date +%Y%m%d_%H%M%S).log"
  record "START reward server: http://127.0.0.1:${REWARD_SERVER_PORT}, log=${log}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/code_reward_server.py" \
    --host 127.0.0.1 \
    --port "${REWARD_SERVER_PORT}" >"${log}" 2>&1 &
  REWARD_SERVER_PID=$!
  trap stop_reward_server EXIT
  sleep 3
}

main() {
  cd "${PROJECT_ROOT}"
  record "EPOCH-CONTINUE RUN START: pid=$$, gpu=${GPU_ID}, python=${PYTHON_BIN}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/prepare_chain_dpo_data.py" | tee -a "${RECORD_MD}"

  run_train_stage "full_dpo_epoch_continue" "${FULL_CONFIG}" "${FULL_OUT}" "${PROJECT_ROOT}/dpo/scripts/train_with_compat.sh" "trainer"
  run_eval_full "full_dpo_epoch_continue" "${FULL_OUT}"

  run_train_stage "lora_dpo_epoch_continue" "${LORA_CONFIG}" "${LORA_OUT}" "${PROJECT_ROOT}/dpo/scripts/train_with_compat.sh" "trainer"
  run_eval_lora "lora_dpo_epoch_continue" "${FULL_OUT}" "${LORA_OUT}"

  start_reward_server
  record "PPO base model: ${PPO_BASE_MODEL}."
  run_train_stage "ppo_reward_fn_epoch_continue" "${PPO_CONFIG}" "${PPO_OUT}" "${PROJECT_ROOT}/dpo/scripts/run_ppo_with_reward_api_monitor.sh" "none"
  local ppo_eval_adapter="${PPO_OUT}"
  if [[ ! -f "${ppo_eval_adapter}/adapter_model.safetensors" ]]; then
    ppo_eval_adapter="$(latest_checkpoint "${PPO_OUT}" || true)"
  fi
  run_eval_lora "ppo_reward_fn_epoch_continue" "${PPO_BASE_MODEL}" "${ppo_eval_adapter}"

  record "EPOCH-CONTINUE RUN COMPLETE."
}

main "$@"
