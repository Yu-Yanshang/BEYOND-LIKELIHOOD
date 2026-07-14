#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${PROJECT_ROOT}/LlamaFactory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/shixun/bin/python}"
CONFIG="${CONFIG:?CONFIG is required}"
GPU_ID="${GPU_ID:-0}"
CUDA13_LIB="${CUDA13_LIB:-/opt/conda/envs/shixun/lib/python3.11/site-packages/nvidia/cu13/lib}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export PYTHONPATH="${PROJECT_ROOT}:${LLAMA_FACTORY_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-130}"

case ":${LD_LIBRARY_PATH:-}:" in
  *":${CUDA13_LIB}:"*) ;;
  *) export LD_LIBRARY_PATH="${CUDA13_LIB}:${LD_LIBRARY_PATH:-}" ;;
esac

"${PYTHON_BIN}" dpo/scripts/llamafactory_dpo_compat.py train "${CONFIG}"
