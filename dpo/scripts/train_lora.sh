#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-dpo/configs/qwen3_06b_code_lora_dpo.yaml}"
GPU_ID="${GPU_ID:-3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -c "import torch; print('project_root=', '${PROJECT_ROOT}'); print('config=', '${CONFIG}'); print('CUDA_VISIBLE_DEVICES=', '${CUDA_VISIBLE_DEVICES}'); print('torch.cuda.is_available()=', torch.cuda.is_available()); print('visible_device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
"${PYTHON_BIN}" -m llamafactory.cli train "${CONFIG}"
