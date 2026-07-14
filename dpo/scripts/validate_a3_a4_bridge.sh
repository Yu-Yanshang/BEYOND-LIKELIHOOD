#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="${A3_V2_DATA_ROOT:-${PROJECT_ROOT}/dpo/data/a3_v2_runs}"
OUTPUT_ROOT="${A3_V2_OUTPUT_ROOT:-${PROJECT_ROOT}/dpo/outputs/a3_v2_runs}"
if [[ -z "${RUN_ID:-}" ]]; then
  if [[ -d "${DATA_ROOT}" ]] && compgen -G "${DATA_ROOT}/*" > /dev/null; then
    export RUN_ID="$(ls -1dt "${DATA_ROOT}"/*/ | head -1 | xargs -n1 basename)"
    echo "RUN_ID was not set; using latest A3 v2 run: ${RUN_ID}"
  else
    echo "RUN_ID is required and no previous A3 v2 run exists." >&2
    exit 2
  fi
fi

DATASET_DIR="${DATA_ROOT}/${RUN_ID}"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_ID}"
PROBE_OUTPUT_DIR="${RUN_OUTPUT_DIR}/a4_probe"
LOG_DIR="${RUN_OUTPUT_DIR}/logs"
CONFIG_PATH="${RUN_OUTPUT_DIR}/a4_probe_config.yaml"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/Qwen3-0.6B}"
mkdir -p "${RUN_OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${DATASET_DIR}/dataset_info.json" ]]; then
  echo "Missing A3 v2 dataset_info.json: ${DATASET_DIR}/dataset_info.json" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing MODEL_PATH directory: ${MODEL_PATH}" >&2
  exit 1
fi

cat > "${CONFIG_PATH}" <<EOF
### model
model_name_or_path: ${MODEL_PATH}
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
pref_loss: sigmoid
pref_beta: 0.1
ref_model: ${MODEL_PATH}

### LoRA
lora_rank: 4
lora_alpha: 8
lora_dropout: 0.0
lora_target: q_proj,v_proj

### dataset
dataset_dir: ${DATASET_DIR}
dataset: code_dpo_a3_v2_high_confidence_train
template: qwen3
enable_thinking: false
cutoff_len: 2048
max_samples: 8
overwrite_cache: true
preprocessing_num_workers: 1
dataloader_num_workers: 0

### output
output_dir: ${PROBE_OUTPUT_DIR}
logging_steps: 1
save_strategy: "no"
plot_loss: false
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 1.0
max_steps: 1
lr_scheduler_type: constant
warmup_ratio: 0.0
max_grad_norm: 1.0
gradient_checkpointing: true
bf16: true
fp16: false
ddp_timeout: 180000000

### eval
eval_strategy: "no"
EOF

cd "${PROJECT_ROOT}"
LOG_FILE="${LOG_DIR}/a4_probe.log"
echo "project_root=${PROJECT_ROOT}"
echo "run_id=${RUN_ID}"
echo "dataset_dir=${DATASET_DIR}"
echo "model_path=${MODEL_PATH}"
echo "config=${CONFIG_PATH}"
echo "log=${LOG_FILE}"

"${PYTHON_BIN}" -c "import torch; print('torch.cuda.is_available()=', torch.cuda.is_available()); print('visible_device_count=', torch.cuda.device_count()); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
"${PYTHON_BIN}" -m llamafactory.cli train "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"

if ! grep -Eq "loss|rewards/|train_loss" "${LOG_FILE}"; then
  echo "A4 bridge probe finished but expected loss/reward markers were not found in ${LOG_FILE}" >&2
  exit 1
fi

echo "A4 bridge probe finished for RUN_ID=${RUN_ID}"
echo "Log: ${LOG_FILE}"
