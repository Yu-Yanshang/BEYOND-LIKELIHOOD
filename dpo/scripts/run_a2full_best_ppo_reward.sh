#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-a2full_best_ppo_reward_$(date +%Y%m%d_%H%M%S)}"

BASE_MODEL="${BASE_MODEL:-/root/project/best_path/runs/best_full_dpo_a5_20260702/models/a2_full}"
REWARD_SERVER_HOST="${REWARD_SERVER_HOST:-127.0.0.1}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-18080}"
REWARD_URL="http://${REWARD_SERVER_HOST}:${REWARD_SERVER_PORT}"

DATASET_NAME="${DATASET_NAME:-a2full_reward_ppo_train}"
DATASET_FILE="${DATASET_FILE:-a2full_reward_ppo_train.json}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
DPO_MIX_LIMIT="${DPO_MIX_LIMIT:-0}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
MAX_STEPS="${MAX_STEPS:-0}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-7}"
PPO_TARGET="${PPO_TARGET:-1.5}"
SAVE_STEPS="${SAVE_STEPS:-50}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_LIMIT="${EVAL_LIMIT:-80}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REQUIRE_ANY_GAIN="${REQUIRE_ANY_GAIN:-1}"
ALLOW_GUARD_FAIL="${ALLOW_GUARD_FAIL:-0}"

RUN_ROOT="${PROJECT_ROOT}/dpo/outputs/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
CONFIG_DIR="${RUN_ROOT}/rendered_configs"
EVAL_DIR="${RUN_ROOT}/mbpp_eval"
PPO_OUT="${RUN_ROOT}/ppo_adapter"
REWARD_LOG="${LOG_DIR}/reward_scores.jsonl"
RECORD_MD="${RUN_ROOT}/TRAINING_RECORD.md"
TEMPLATE="${PROJECT_ROOT}/dpo/configs/a2full_best_ppo_reward_fn.yaml"
RENDERED_CONFIG="${CONFIG_DIR}/ppo_reward.yaml"

REWARD_SERVER_PID=""

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PROJECT_ROOT}/dpo/scripts:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-130}"
CUDA13_LIB="${CUDA13_LIB:-/opt/conda/envs/shixun/lib/python3.11/site-packages/nvidia/cu13/lib}"
case ":${LD_LIBRARY_PATH:-}:" in
  *":${CUDA13_LIB}:"*) ;;
  *) export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}" ;;
esac

record() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*" | tee -a "${RECORD_MD}"
}

stop_reward_server() {
  if [[ -n "${REWARD_SERVER_PID}" ]]; then
    kill "${REWARD_SERVER_PID}" 2>/dev/null || true
    wait "${REWARD_SERVER_PID}" 2>/dev/null || true
  fi
}

wait_for_reward_server() {
  local url="$1"
  for _ in $(seq 1 20); do
    if "${PYTHON_BIN}" -c "import urllib.request; urllib.request.urlopen('${url}', timeout=1)" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_reward_server() {
  local server_log="${LOG_DIR}/reward_server.log"
  record "START reward server: ${REWARD_URL}, log=${server_log}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/a2full_ppo_reward_server.py" \
    --host "${REWARD_SERVER_HOST}" \
    --port "${REWARD_SERVER_PORT}" \
    --enable-prompt-tests \
    --test-timeout "${REWARD_TEST_TIMEOUT:-1.5}" \
    --memory-mb "${REWARD_MEMORY_MB:-512}" \
    --log-jsonl "${REWARD_LOG}" >"${server_log}" 2>&1 &
  REWARD_SERVER_PID=$!
  trap stop_reward_server EXIT
  wait_for_reward_server "${REWARD_URL}" || {
    tail -80 "${server_log}" >&2 || true
    echo "Reward server failed to become ready: ${REWARD_URL}" >&2
    exit 1
  }
}

latest_checkpoint() {
  local dir="$1"
  if [[ -d "${dir}" ]]; then
    find "${dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
  fi
}

ppo_adapter_for_eval() {
  if [[ -f "${PPO_OUT}/adapter_model.safetensors" ]]; then
    printf '%s\n' "${PPO_OUT}"
    return 0
  fi
  latest_checkpoint "${PPO_OUT}"
}

init_record() {
  mkdir -p "${LOG_DIR}" "${CONFIG_DIR}" "${EVAL_DIR}" "${PPO_OUT}"
  if [[ ! -f "${RECORD_MD}" ]]; then
    cat >"${RECORD_MD}" <<EOF
# A2-full PPO reward-function run

- Run ID: ${RUN_ID}
- Base policy: ${BASE_MODEL}
- Reference model for PPO KL: ${BASE_MODEL}
- Reward API: ${REWARD_URL}
- PPO adapter output: ${PPO_OUT}
- Dataset: ${DATASET_NAME}
- Loss design: TRL/LLaMA-Factory PPO clipped policy loss + value loss + adaptive KL to the A2-full reference.
- Anti-forgetting controls: LoRA-only q/k/v/o updates, low LR, reward clipping, score normalization, reward whitening, low PPO KL target, MBPP baseline guard.
- Output controls: reward penalizes thinking tags, chat special tokens, explanations, copied asserts, sample prints, repetition, and overlong/truncated code.

## Timeline
EOF
  fi
}

prepare_data() {
  record "PREPARE PPO data: dataset=${DATASET_NAME}, max_train_samples=${MAX_TRAIN_SAMPLES}, dpo_mix_limit=${DPO_MIX_LIMIT}."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/prepare_a2full_ppo_data.py" \
    --dataset-name "${DATASET_NAME}" \
    --output "${DATASET_FILE}" \
    --limit "${MAX_TRAIN_SAMPLES}" \
    --dpo-limit "${DPO_MIX_LIMIT}" 2>&1 | tee "${LOG_DIR}/prepare_data.log"
}

render_config() {
  record "RENDER PPO config: ${RENDERED_CONFIG}."
  args=(
    --template "${TEMPLATE}"
    --output "${RENDERED_CONFIG}"
    --model-path "${BASE_MODEL}"
    --reward-url "${REWARD_URL}"
    --output-dir "./dpo/outputs/${RUN_ID}/ppo_adapter"
    --dataset "${DATASET_NAME}"
    --num-train-epochs "${NUM_TRAIN_EPOCHS}"
    --learning-rate "${LEARNING_RATE}"
    --ppo-target "${PPO_TARGET}"
    --save-steps "${SAVE_STEPS}"
  )
  if [[ "${MAX_STEPS}" != "0" ]]; then
    args+=(--max-steps "${MAX_STEPS}")
  fi
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/render_a2full_ppo_config.py" "${args[@]}"
}

run_train() {
  if [[ "${RUN_TRAIN}" != "1" ]]; then
    record "SKIP training because RUN_TRAIN=${RUN_TRAIN}."
    return
  fi
  if [[ -f "${PPO_OUT}/adapter_model.safetensors" ]]; then
    record "SKIP training: final adapter already exists at ${PPO_OUT}."
    return
  fi
  record "START PPO training."
  "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/llamafactory_dpo_compat.py" train "${RENDERED_CONFIG}" \
    2>&1 | tee "${LOG_DIR}/ppo_train.log"
  record "END PPO training."
}

eval_args() {
  if [[ "${EVAL_LIMIT}" != "0" ]]; then
    printf -- '--limit %q' "${EVAL_LIMIT}"
  fi
}

run_eval_and_guard() {
  if [[ "${RUN_EVAL}" != "1" ]]; then
    record "SKIP eval because RUN_EVAL=${RUN_EVAL}."
    return
  fi

  local adapter
  adapter="$(ppo_adapter_for_eval || true)"
  if [[ -z "${adapter}" ]]; then
    echo "No PPO adapter/checkpoint found for evaluation under ${PPO_OUT}" >&2
    exit 1
  fi

  local base_eval="${EVAL_DIR}/a2_full"
  local ppo_eval="${EVAL_DIR}/ppo_reward"
  mkdir -p "${base_eval}" "${ppo_eval}"

  record "EVAL baseline A2-full on MBPP: output=${base_eval}, limit=${EVAL_LIMIT}."
  base_cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/mbpp_eval_dpo.py"
    --config sanitized
    --split test
    --prompt_mode zero_shot
    --model_path "${BASE_MODEL}"
    --output_dir "${base_eval}"
    --batch_size "${BATCH_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
  )
  if [[ "${EVAL_LIMIT}" != "0" ]]; then
    base_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  "${base_cmd[@]}" 2>&1 | tee "${LOG_DIR}/eval_a2_full.log"

  record "EVAL PPO adapter on MBPP: adapter=${adapter}, output=${ppo_eval}, limit=${EVAL_LIMIT}."
  ppo_cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/mbpp_eval_lora_dpo.py"
    --config sanitized
    --split test
    --prompt_mode zero_shot
    --model_path "${BASE_MODEL}"
    --adapter_path "${adapter}"
    --output_dir "${ppo_eval}"
    --batch_size "${BATCH_SIZE}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
  )
  if [[ "${EVAL_LIMIT}" != "0" ]]; then
    ppo_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  "${ppo_cmd[@]}" 2>&1 | tee "${LOG_DIR}/eval_ppo_reward.log"

  record "COMPARE PPO against A2-full guard thresholds."
  compare_cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/dpo/scripts/compare_a2full_ppo_metrics.py"
    --base-metrics "${base_eval}/mbpp_metrics.json"
    --ppo-metrics "${ppo_eval}/mbpp_metrics.json"
    --output "${RUN_ROOT}/ppo_guard_report.json"
    --max-pass-drop "${MAX_PASS_DROP:-0.02}"
    --max-test-drop "${MAX_TEST_DROP:-0.03}"
    --max-syntax-drop "${MAX_SYNTAX_DROP:-0.05}"
  )
  if [[ "${REQUIRE_ANY_GAIN}" == "1" ]]; then
    compare_cmd+=(--require-any-gain)
  fi
  set +e
  "${compare_cmd[@]}" 2>&1 | tee "${LOG_DIR}/guard_compare.log"
  local guard_status=$?
  set -e
  if [[ "${guard_status}" != "0" ]]; then
    record "GUARD FAILED with status=${guard_status}; see ${RUN_ROOT}/ppo_guard_report.json."
    if [[ "${ALLOW_GUARD_FAIL}" != "1" ]]; then
      exit "${guard_status}"
    fi
  else
    record "GUARD PASSED: ${RUN_ROOT}/ppo_guard_report.json."
  fi
}

main() {
  cd "${PROJECT_ROOT}"
  init_record
  record "RUN START: pid=$$, gpu=${GPU_ID}, python=${PYTHON_BIN}."
  prepare_data
  render_config
  start_reward_server
  run_train
  run_eval_and_guard
  record "RUN COMPLETE."
}

main "$@"
