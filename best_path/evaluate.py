#!/usr/bin/env python3
"""Unified execution-based MBPP evaluator for greedy and A5 enhanced inference."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/root/project")
DEFAULT_MBPP = PROJECT_ROOT.parent / "mbpp/sanitized/test-00000-of-00001.parquet"
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.I | re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_MBPP)
    parser.add_argument("--strategy", choices=("greedy", "enhanced"), default="greedy")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--repairs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-timeout", type=float, default=5.0)
    parser.add_argument("--memory-mb", type=int, default=1024)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def prompt_text(row: dict[str, Any]) -> str:
    tests = "\n".join(listify(row.get("test_list")))
    return (
        "Solve the Python programming task below. Think briefly about edge cases, then return one complete solution "
        "inside a ```python code block. Do not use external input/output unless requested.\n\n"
        f"Task:\n{str(row.get('prompt', '')).strip()}\n\nVisible tests:\n{tests}"
    )


def repair_prompt(row: dict[str, Any], code: str, failures: list[dict[str, Any]]) -> str:
    details = []
    tests = listify(row.get("test_list"))
    for index, result in enumerate(failures):
        if not result.get("passed"):
            error = str(result.get("stderr", ""))[-600:]
            details.append(f"Test: {tests[index]}\nFailure: {error or result.get('error_type', 'failed')}")
    return (
        f"{prompt_text(row)}\n\nThe following candidate is incorrect:\n```python\n{code}\n```\n\n"
        "Diagnose the failures and return a corrected complete solution in one ```python block.\n" + "\n\n".join(details[:3])
    )


def render(tokenizer: Any, user: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise Python code-generation assistant."},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f"System: {messages[0]['content']}\nUser: {user}\nAssistant:"


def extract_code(text: str) -> str:
    text = str(text or "").strip()
    blocks = FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "class ", "import ", "from ", "@")):
            start = index
            break
    return "\n".join(line for line in lines[start:] if not line.strip().startswith("```")).strip()


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def resource_limit(memory_mb: int, timeout: float) -> None:
    try:
        import resource
        memory = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout) + 1),) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
    except Exception:
        pass


def run_test(code: str, setup: str, assertion: str, args: argparse.Namespace) -> dict[str, Any]:
    source = "\n\n".join(x for x in ("import faulthandler\nfaulthandler.enable()", setup, code, assertion) if x.strip())
    with tempfile.TemporaryDirectory(prefix="best_path_mbpp_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(source + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["HOME"] = tmp
        try:
            result = subprocess.run(
                [sys.executable, str(path)], cwd=tmp, env=env, text=True, capture_output=True,
                timeout=args.test_timeout,
                preexec_fn=lambda: resource_limit(args.memory_mb, args.test_timeout) if os.name == "posix" else None,
            )
            return {
                "passed": result.returncode == 0,
                "error_type": "" if result.returncode == 0 else "runtime_error",
                "stderr": result.stderr[-1000:],
            }
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error_type": "timeout", "stderr": str(exc)}


def score(code: str, row: dict[str, Any], args: argparse.Namespace, source: str, response: str) -> dict[str, Any]:
    setup = "\n".join(listify(row.get("test_imports")))
    tests = listify(row.get("test_list"))
    results = [run_test(code, setup, test, args) for test in tests]
    return {
        "source": source,
        "response": response,
        "code": code,
        "syntax_ok": syntax_ok(code),
        "passed_tests": sum(bool(item["passed"]) for item in results),
        "total_tests": len(tests),
        "passed": bool(tests) and all(item["passed"] for item in results),
        "test_results": results,
    }


def normalize_for_consensus(code: str) -> str:
    try:
        return ast.dump(ast.parse(code), annotate_fields=False, include_attributes=False)
    except SyntaxError:
        return " ".join(code.split())


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_for_consensus(item["code"]) for item in candidates]
    signatures = [(item["passed_tests"], item["syntax_ok"], norm[:240]) for item, norm in zip(candidates, normalized)]
    votes = Counter(signatures)
    for index, item in enumerate(candidates):
        item["consensus_votes"] = votes[signatures[index]]
        item["mean_similarity"] = sum(
            SequenceMatcher(None, normalized[index][:2000], other[:2000]).ratio() for other in normalized
        ) / max(1, len(normalized))
    return max(
        candidates,
        key=lambda item: (
            item["passed_tests"], item["syntax_ok"], item["consensus_votes"], item["mean_similarity"], -len(item["code"])
        ),
    )


def generate(model: Any, tokenizer: Any, torch: Any, prompts: list[str], repeats: int, args: argparse.Namespace, sample: bool) -> list[list[str]]:
    expanded = [render(tokenizer, prompt) for prompt in prompts for _ in range(repeats)]
    inputs = tokenizer(expanded, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    response_ids = output[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    return [decoded[index:index + repeats] for index in range(0, len(decoded), repeats)]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input_file)
    if args.limit:
        rows = rows[:args.limit]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True,
    )
    model.eval()
    started = time.time()
    cases: list[dict[str, Any]] = []
    repeats = 1 if args.strategy == "greedy" else args.candidates
    sample = args.strategy == "enhanced"

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        outputs = generate(model, tokenizer, torch, [prompt_text(row) for row in batch], repeats, args, sample)
        for row, responses in zip(batch, outputs, strict=True):
            candidates = [score(extract_code(response), row, args, "greedy" if not sample else "best_of_n", response) for response in responses]
            initial = select_candidate(candidates)
            if args.strategy == "enhanced" and not initial["passed"] and args.repairs > 0:
                repairs = generate(
                    model, tokenizer, torch,
                    [repair_prompt(row, initial["code"], initial["test_results"])], args.repairs, args, True,
                )[0]
                candidates.extend(score(extract_code(response), row, args, "reflexion", response) for response in repairs)
            selected = select_candidate(candidates)
            cases.append({
                "task_id": int(row["task_id"]),
                "prompt": str(row.get("prompt", "")),
                "selected": selected,
                "initial_best_passed": initial["passed"],
                "candidate_count": len(candidates),
                "all_candidates": candidates,
            })
        print(f"evaluated {len(cases)}/{len(rows)}", flush=True)

    elapsed = time.time() - started
    total = len(cases)
    passed = sum(case["selected"]["passed"] for case in cases)
    syntax = sum(case["selected"]["syntax_ok"] for case in cases)
    passed_tests = sum(case["selected"]["passed_tests"] for case in cases)
    total_tests = sum(case["selected"]["total_tests"] for case in cases)
    initial_passed = sum(case["initial_best_passed"] for case in cases)
    reflexion_selected = sum(case["selected"]["source"] == "reflexion" for case in cases)
    metrics = {
        "benchmark": "MBPP sanitized",
        "model_path": str(args.model_path),
        "strategy": args.strategy,
        "enable_thinking": False,
        "total": total,
        "pass_at_1": passed / total if total else 0.0,
        "syntax_pass_rate": syntax / total if total else 0.0,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "passed_tasks": passed,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "initial_best_of_n_passed_tasks": initial_passed,
        "reflexion_selected_tasks": reflexion_selected,
        "candidates": repeats,
        "max_repairs": args.repairs if args.strategy == "enhanced" else 0,
        "runtime_seconds": elapsed,
        "average_latency_seconds": elapsed / total if total else None,
        "protocol": "Same visible MBPP tests are used in the prompt and the sandboxed execution verifier.",
    }
    save_json(args.output_dir / "metrics.json", metrics)
    save_jsonl(args.output_dir / "cases.jsonl", cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
