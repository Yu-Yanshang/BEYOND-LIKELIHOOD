#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

bash dpo/scripts/prepare_data.sh
bash dpo/scripts/train_lora.sh
bash dpo/scripts/test_code_lora_dpo.sh
