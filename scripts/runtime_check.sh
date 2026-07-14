#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/project}"
source "${PROJECT_ROOT}/scripts/common_env.sh"
cd "${PROJECT_ROOT}"

echo "== project =="
pwd
test -d Qwen3-0.6B
test -f Qwen3-0.6B/config.json
test -f Qwen3-0.6B/model.safetensors
test -d LlamaFactory/src/llamafactory
test -f sft/configs/qwen3_06b_code_full_sft.yaml
test -f dpo/configs/qwen3_06b_code_full_dpo.yaml

echo
echo "== gpu =="
nvidia-smi | sed -n '1,24p' || true

echo
echo "== python/runtime =="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

print("python", sys.executable)
for name in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets", "pandas", "pyarrow", "yaml", "llamafactory"]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            print(name, "MISSING")
            continue
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print(name, "ERR", repr(exc))

import torch
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("device0", props.name, "mem_gib", round(props.total_memory / 1024**3, 2))

root = Path("/root/project")
for rel in ["sft/data/code_sft_train.json", "sft/data/code_sft_valid.json", "dpo/data/code_dpo_train.json", "dpo/data/code_dpo_test.json", "sft/data/mbpp_sanitized_test.json"]:
    path = root / rel
    if not path.exists():
        print(rel, "MISSING")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    print(rel, "count", len(data), "first_keys", list(data[0].keys()) if data else [])
PY

echo
echo "== qwen3 tokenizer/config =="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

model_path = Path("/root/project/Qwen3-0.6B")
cfg = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
print("model_type", cfg.model_type)
print("architectures", getattr(cfg, "architectures", None))
print("hidden_size", getattr(cfg, "hidden_size", None), "layers", getattr(cfg, "num_hidden_layers", None))
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
messages = [{"role": "user", "content": "Write a Python function that returns x + 1."}]
try:
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("chat_template_ok", bool(rendered), "rendered_prefix", rendered[:160].replace("\n", "\\n"))
PY

if [[ "${RUN_QWEN3_GPU_FORWARD:-0}" == "1" ]]; then
  echo
  echo "== optional qwen3 gpu forward =="
  "${PYTHON_BIN}" - <<'PY'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/root/project/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
model.eval()
inputs = tok("print hello", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs, use_cache=False)
print("forward_ok", tuple(out.logits.shape), "peak_mem_gib", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
PY
else
  echo
  echo "optional GPU forward skipped; run with RUN_QWEN3_GPU_FORWARD=1 to expose vGPU/model execution issues"
fi

echo
echo "runtime_check=PASS"
