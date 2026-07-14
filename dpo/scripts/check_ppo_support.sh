#!/usr/bin/env bash
set -euo pipefail

LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR:-/root/project/LlamaFactory}
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ ! -d "${LLAMA_FACTORY_DIR}" ]]; then
  echo "Missing LLaMA-Factory directory: ${LLAMA_FACTORY_DIR}" >&2
  exit 1
fi

cd "${LLAMA_FACTORY_DIR}"

echo "[1/4] Checking PPO source directory"
test -d src/llamafactory/train/ppo && echo "OK: src/llamafactory/train/ppo exists"

echo "[2/4] Checking train dispatcher"
grep -R "run_ppo\|stage.*ppo\|Unknown task" -n src/llamafactory/train/tuner.py src/llamafactory/train/ppo | head -80

echo "[3/4] Checking PPO/RM hparams"
grep -R "ppo_buffer_size\|ppo_epochs\|reward_model\|reward_model_type" -n src/llamafactory/hparams src/llamafactory/train | head -120

echo "[4/4] Checking Python imports and TRL version"
PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import pathlib
import llamafactory
print("llamafactory:", pathlib.Path(llamafactory.__file__).resolve())
try:
    import trl
    print("trl:", trl.__version__)
except Exception as exc:
    print("trl import failed:", repr(exc))
PY

echo "PPO support check finished. If run_ppo and src/llamafactory/train/ppo are present, this source tree supports PPO."
