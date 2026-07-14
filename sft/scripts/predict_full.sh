#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
CONFIG="${CONFIG:-sft/configs/qwen3_06b_code_full_predict.yaml}"
GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${PROJECT_ROOT}"

if [ -n "${MODEL_PATH}" ]; then
  echo "Using model path: ${MODEL_PATH}"
  TMP_CONFIG="$(mktemp "${PROJECT_ROOT}/sft/outputs/.predict_config.XXXXXX.yaml")"
  "${PYTHON_BIN}" - "$CONFIG" "$MODEL_PATH" "$TMP_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path, model_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
config_path = Path(config_path)
out_path = Path(out_path)

with config_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

data["model_name_or_path"] = model_path
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Wrote override config to {out_path}")
PY
  CONFIG="$TMP_CONFIG"
fi

"${PYTHON_BIN}" -c "import torch; print('project_root=', '${PROJECT_ROOT}'); print('CUDA_VISIBLE_DEVICES=', '${CUDA_VISIBLE_DEVICES}'); print('torch.cuda.is_available()=', torch.cuda.is_available()); print('visible_device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
"${PYTHON_BIN}" -m llamafactory.cli train "${CONFIG}"
