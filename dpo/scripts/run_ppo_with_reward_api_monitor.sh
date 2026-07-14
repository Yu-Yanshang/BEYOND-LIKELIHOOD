#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
CONFIG="${CONFIG:-dpo/configs/a2full_chain_ppo_reward_fn.yaml}"
GPU_ID="${GPU_ID:-0}"
PPO_PATIENCE_EPOCHS="${PPO_PATIENCE_EPOCHS:-1}"
PPO_CHECK_INTERVAL="${PPO_CHECK_INTERVAL:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
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

OUTPUT_DIR="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${CONFIG}', encoding='utf-8'))['output_dir'])")"
TRAINER_LOG="${PROJECT_ROOT}/${OUTPUT_DIR#./}/trainer_log.jsonl"
mkdir -p "${PROJECT_ROOT}/${OUTPUT_DIR#./}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}" \
"${PYTHON_BIN}" dpo/scripts/llamafactory_dpo_compat.py train "${CONFIG}" &
TRAIN_PID=$!

best_epoch_reward=""
bad_epochs=0
last_completed_epoch=0

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep "${PPO_CHECK_INTERVAL}"
  if [[ ! -s "${TRAINER_LOG}" ]]; then
    continue
  fi

  status="$("${PYTHON_BIN}" - "${TRAINER_LOG}" <<'PY'
import json
import math
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(row.get("epoch"), (int, float)) and isinstance(row.get("reward"), (int, float)):
        rows.append((float(row["epoch"]), float(row["reward"])))

if not rows:
    raise SystemExit(0)

max_epoch = max(epoch for epoch, _ in rows)
completed_epoch = int(math.floor(max_epoch + 1e-9))
if completed_epoch < 1:
    raise SystemExit(0)

lower = completed_epoch - 1
rewards = [reward for epoch, reward in rows if lower < epoch <= completed_epoch]
if not rewards:
    raise SystemExit(0)

avg_reward = sum(rewards) / len(rewards)
print(f"{completed_epoch} {avg_reward:.10f} {len(rewards)}")
PY
)"
  if [[ -z "${status}" ]]; then
    continue
  fi

  read -r completed_epoch epoch_reward reward_count <<<"${status}"
  if (( completed_epoch <= last_completed_epoch )); then
    continue
  fi
  last_completed_epoch="${completed_epoch}"

  echo "PPO epoch ${completed_epoch} reward_avg=${epoch_reward} reward_logs=${reward_count}"
  improved="$("${PYTHON_BIN}" -c "import sys; best=sys.argv[1]; cur=float(sys.argv[2]); print('1' if best == '' or cur > float(best) else '0')" "${best_epoch_reward}" "${epoch_reward}")"
  if [[ "${improved}" == "1" ]]; then
    best_epoch_reward="${epoch_reward}"
    bad_epochs=0
  else
    bad_epochs=$((bad_epochs + 1))
  fi

  if (( bad_epochs >= PPO_PATIENCE_EPOCHS )); then
    echo "PPO early stop: average reward did not improve for ${PPO_PATIENCE_EPOCHS} completed epoch(s). best_epoch_reward=${best_epoch_reward}"
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
    exit 0
  fi
done

wait "${TRAIN_PID}"
