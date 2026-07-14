#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/sft/a2_v2/common_env.sh"

RUN_ID="${RUN_ID:?RUN_ID is required}"
A2_OUTPUT="${PROJECT_ROOT}/sft/outputs/a2_v2_runs/${RUN_ID}"
HANDOFF="${A2_OUTPUT}/a2_handoff_manifest.json"
[[ -f "${HANDOFF}" ]] || { echo "Missing A2 handoff: ${HANDOFF}" >&2; exit 1; }

if [[ -z "${A3_RUN_ID:-}" ]]; then
  latest="$(find "${PROJECT_ROOT}/dpo/data/a3_v2_runs" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
  A3_RUN_ID="${latest}"
fi
[[ -n "${A3_RUN_ID}" ]] || { echo "A3_RUN_ID is required and no A3 v2 run exists" >&2; exit 1; }
A3_DATA="${PROJECT_ROOT}/dpo/data/a3_v2_runs/${A3_RUN_ID}"
[[ -f "${A3_DATA}/dataset_info.json" ]] || { echo "Missing A3 registry: ${A3_DATA}" >&2; exit 1; }

BRIDGE_ROOT="${A2_OUTPUT}/a4_bridge"
mkdir -p "${BRIDGE_ROOT}"
SUMMARY="${BRIDGE_ROOT}/bridge_summary.json"

for variant in full qlora qdora; do
  MODEL_PATH="$(${PYTHON_BIN} - "${HANDOFF}" "${variant}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))["variants"][sys.argv[2]]
print(x["merged_model_path"])
PY
)"
  [[ -d "${MODEL_PATH}" ]] || { echo "Missing ${variant} handoff model: ${MODEL_PATH}" >&2; exit 1; }
  OUT="${BRIDGE_ROOT}/${variant}"
  CFG="${BRIDGE_ROOT}/${variant}.yaml"
  "${PYTHON_BIN}" - "${CFG}" "${MODEL_PATH}" "${A3_DATA}" "${OUT}" <<'PY'
import sys,yaml
path,model,data,out=sys.argv[1:]
config={
 "model_name_or_path":model,"trust_remote_code":True,"stage":"dpo","do_train":True,
 "finetuning_type":"lora","pref_loss":"sigmoid","pref_beta":0.1,"ref_model":model,
 "lora_rank":4,"lora_alpha":8,"lora_dropout":0.0,"lora_target":"q_proj,v_proj",
 "dataset_dir":data,"dataset":"code_dpo_a3_v2_high_confidence_train","template":"qwen3",
 "enable_thinking":False,"cutoff_len":2048,"max_samples":8,"overwrite_cache":True,
 "preprocessing_num_workers":1,"dataloader_num_workers":0,"output_dir":out,"logging_steps":1,
 "save_strategy":"no","plot_loss":False,"overwrite_output_dir":True,"report_to":"none",
 "per_device_train_batch_size":1,"gradient_accumulation_steps":1,"learning_rate":1e-4,
 "max_steps":1,"num_train_epochs":1.0,"lr_scheduler_type":"constant","warmup_ratio":0.0,
 "gradient_checkpointing":True,"bf16":True,"fp16":False,"eval_strategy":"no"}
open(path,"w",encoding="utf-8").write(yaml.safe_dump(config,sort_keys=False))
PY
  "${PYTHON_BIN}" -m sft.a2_v2.llamafactory_compat train "${CFG}" 2>&1 | tee "${BRIDGE_ROOT}/${variant}.log"
  grep -Eq 'loss|rewards/|train_loss' "${BRIDGE_ROOT}/${variant}.log" || { echo "Missing DPO metrics for ${variant}" >&2; exit 1; }
done

"${PYTHON_BIN}" - "${SUMMARY}" "${RUN_ID}" "${A3_RUN_ID}" <<'PY'
import json,sys
from datetime import datetime,timezone
out={"created_at":datetime.now(timezone.utc).isoformat(),"a2_run_id":sys.argv[2],"a3_run_id":sys.argv[3],
     "dataset":"code_dpo_a3_v2_high_confidence_train","max_steps":1,
     "variants":{x:{"status":"passed"} for x in ("full","qlora","qdora")}}
open(sys.argv[1],"w").write(json.dumps(out,indent=2)+"\n")
PY
echo "A2→A3→A4 bridge passed: ${SUMMARY}"
