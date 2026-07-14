#!/usr/bin/env bash
set -euo pipefail

# End-to-end PPO alignment:
# 1) build prompt-only PPO data from code_dpo_train.json
# 2) train LoRA reward model on code_dpo_train ranking data
# 3) train LoRA policy with PPO on code_ppo_train prompts
# 4) run static code-generation comparison against base

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-0}

cd "${PROJECT_ROOT}"

bash dpo/scripts/prepare_ppo_data.sh
GPU_ID="${GPU_ID}" PYTHON_BIN="${PYTHON_BIN}" bash dpo/scripts/train_reward_model.sh
GPU_ID="${GPU_ID}" PYTHON_BIN="${PYTHON_BIN}" bash dpo/scripts/train_ppo.sh
"${PYTHON_BIN}" dpo/scripts/test_code_ppo.py --gpu_id "${GPU_ID}"
