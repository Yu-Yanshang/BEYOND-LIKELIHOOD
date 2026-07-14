from pathlib import Path
import subprocess
import sys
import py_compile

REPO = Path("/root/project/LlamaFactory")
REL = Path("src/llamafactory/train/rm/trainer.py")
PATH = REPO / REL

def run(cmd):
    return subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def compile_check(stage):
    try:
        py_compile.compile(str(PATH), doraise=True)
        print(f"[OK] py_compile passed after {stage}")
    except Exception as e:
        print(f"[ERROR] py_compile failed after {stage}: {e}")
        print("\nPlease inspect:")
        print(f"  nl -ba {PATH} | sed -n '120,190p'")
        sys.exit(1)

if not PATH.exists():
    print(f"[ERROR] file not found: {PATH}")
    sys.exit(1)

# Step 1: restore original file from git, avoiding repeated broken patches.
restored = False
for cmd in [
    ["git", "restore", str(REL)],
    ["git", "checkout", "--", str(REL)],
]:
    r = run(cmd)
    if r.returncode == 0:
        print(f"[OK] restored original file by: {' '.join(cmd)}")
        restored = True
        break
    else:
        print(f"[WARN] failed: {' '.join(cmd)}")
        print(r.stderr.strip())

if not restored:
    print("[ERROR] Could not restore trainer.py from git.")
    print("This means the repository may not have git metadata or the file is not tracked.")
    print("Recommended recovery:")
    print("  cd /root/project")
    print("  mv LlamaFactory LlamaFactory_broken")
    print("  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git LlamaFactory")
    print("  cd LlamaFactory")
    print("  python -m pip install -e \".[torch,metrics]\"")
    sys.exit(1)

compile_check("git restore")

text = PATH.read_text(encoding="utf-8")
lines = text.splitlines()

target = "super()._save(output_dir, state_dict)"
target_indices = [i for i, line in enumerate(lines) if target in line]

if len(target_indices) != 1:
    print(f"[ERROR] expected exactly one target line, found {len(target_indices)}")
    for i in target_indices:
        print(f"line {i + 1}: {lines[i]}")
    sys.exit(1)

idx = target_indices[0]
target_line = lines[idx]
indent = target_line[: len(target_line) - len(target_line.lstrip())]

block = [
    indent + "# BEGIN_FIX_SHARED_TENSORS_FOR_SAFETENSORS",
    indent + "# Some causal LMs tie input embeddings and lm_head weights.",
    indent + "# safetensors cannot directly serialize tensors sharing the same memory.",
    indent + "# Clone duplicated tensor references in the state_dict before saving.",
    indent + "if state_dict is not None:",
    indent + "    _seen_tensor_ptrs = {}",
    indent + "    for _name, _tensor in list(state_dict.items()):",
    indent + "        if hasattr(_tensor, 'data_ptr'):",
    indent + "            try:",
    indent + "                _ptr = _tensor.data_ptr()",
    indent + "            except Exception:",
    indent + "                _ptr = None",
    indent + "            if _ptr is not None:",
    indent + "                if _ptr in _seen_tensor_ptrs:",
    indent + "                    state_dict[_name] = _tensor.clone()",
    indent + "                else:",
    indent + "                    _seen_tensor_ptrs[_ptr] = _name",
    indent + "# END_FIX_SHARED_TENSORS_FOR_SAFETENSORS",
]

new_lines = lines[:idx] + block + [target_line] + lines[idx + 1:]
PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

compile_check("patch insertion")

print("[OK] patched successfully:", PATH)
print("[INFO] Patched region:")
start = max(0, idx - 5)
end = min(len(new_lines), idx + len(block) + 6)
for no in range(start, end):
    print(f"{no + 1:04d}: {new_lines[no]}")
