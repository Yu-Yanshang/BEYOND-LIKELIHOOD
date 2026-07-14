#!/usr/bin/env python3
"""Evaluate a model on Python code tasks with the CoT prompt template."""

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
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from cot_code_example import build_chat_messages, extract_final_code, normalize_text
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from cot_code_example import build_chat_messages, extract_final_code, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "dpo" / "outputs" / "qwen3_06b_code_full_dpo"
DEFAULT_INPUT = WORK_ROOT / "mbpp" / "sanitized" / "test-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tts" / "outputs" / "cot_code_eval"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--test_timeout", type=float, default=5.0)
    parser.add_argument("--memory_mb", type=int, default=1024)
    return parser.parse_args()


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except Exception:
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict("records")
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet file: {path}") from exc


def load_rows(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if path.suffix == ".parquet":
        return read_parquet(path)
    return [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = normalize_text(row.get(key))
        if value:
            return value
    return ""


def row_to_task(row: Any) -> str:
    if isinstance(row, str):
        return normalize_text(row)
    if not isinstance(row, dict):
        return normalize_text(row)

    prompt = first_text(row, ("prompt", "instruction", "question", "task", "text"))
    input_text = first_text(row, ("input", "query"))
    if prompt and input_text:
        task = f"{prompt}\n\n{input_text}"
    else:
        task = prompt or input_text

    tests = listify(row.get("test_list"))
    if tests:
        task = f"{task}\n\nYour code should pass these tests:\n" + "\n".join(tests)
    return task


def row_to_reference(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return first_text(row, ("output", "chosen", "answer", "reference", "label", "code"))


def row_to_tests(row: Any) -> tuple[str, list[str]]:
    if not isinstance(row, dict):
        return "", []
    setup_parts = []
    setup_parts.extend(listify(row.get("test_imports")))
    setup = first_text(row, ("test_setup_code", "setup_code"))
    if setup:
        setup_parts.append(setup)
    tests = listify(row.get("test_list"))
    tests.extend(listify(row.get("challenge_test_list")))
    return "\n".join(setup_parts), tests


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def code_tokens(code: str) -> list[str]:
    return TOKEN_RE.findall(code)


def token_f1(prediction: str, reference: str) -> float | None:
    if not reference:
        return None
    pred_tokens = code_tokens(prediction)
    ref_tokens = code_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def format_chat(tokenizer: Any, task: str) -> str:
    messages = build_chat_messages(task)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    system = messages[0]["content"]
    user = messages[1]["content"]
    return f"{system}\n\nUser:\n{user}\n\nAssistant:\n"


def model_input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def generate_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing model directory: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [format_chat(tokenizer, row["task"]) for row in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model_input_device(model))
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature if args.temperature > 0 else None,
            "top_p": args.top_p,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        response_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        outputs.extend(normalize_text(text) for text in tokenizer.batch_decode(response_ids, skip_special_tokens=True))
        print(f"generated {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def limit_resources(memory_mb: int, timeout: float) -> None:
    try:
        import resource

        memory_bytes = memory_mb * 1024 * 1024
        cpu_seconds = max(1, int(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except Exception:
        return


def run_one_assert(code: str, setup: str, test: str, args: argparse.Namespace) -> dict[str, Any]:
    runner = "\n\n".join(part for part in ("import faulthandler\nfaulthandler.enable()", setup, code, test) if part.strip())
    with tempfile.TemporaryDirectory(prefix="cot_code_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["HOME"] = tmpdir
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=args.test_timeout,
                preexec_fn=lambda: limit_resources(args.memory_mb, args.test_timeout) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error_type": "timeout", "stderr": str(exc)}
    return {
        "passed": result.returncode == 0,
        "error_type": "" if result.returncode == 0 else "runtime_error",
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-2000:],
    }


def score_case(row: dict[str, Any], response: str, args: argparse.Namespace) -> dict[str, Any]:
    code = extract_final_code(response)
    reference_code = extract_final_code(row["reference"])
    setup = row["test_setup"]
    tests = row["tests"]
    test_results = [run_one_assert(code, setup, test, args) for test in tests]
    passed_tests = sum(1 for item in test_results if item["passed"])
    exact_match = code == reference_code if reference_code else None

    return {
        "index": row["index"],
        "task": row["task"],
        "reference": row["reference"],
        "response": response,
        "final_code": code,
        "reference_code": reference_code,
        "syntax_ok": syntax_ok(code),
        "exact_match": exact_match,
        "token_f1": token_f1(code, reference_code),
        "passed": passed_tests == len(tests) and len(tests) > 0,
        "passed_tests": passed_tests,
        "total_tests": len(tests),
        "test_results": test_results,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(cases)
    syntax_pass = sum(1 for case in cases if case["syntax_ok"])
    with_tests = [case for case in cases if case["total_tests"] > 0]
    with_exact = [case for case in cases if case["exact_match"] is not None]
    f1_values = [case["token_f1"] for case in cases if case["token_f1"] is not None]
    passed_tasks = sum(1 for case in with_tests if case["passed"])
    total_tests = sum(case["total_tests"] for case in with_tests)
    passed_tests = sum(case["passed_tests"] for case in with_tests)
    return {
        "model_path": str(args.model_path),
        "input_file": str(args.input_file),
        "total": total,
        "syntax_pass_rate": syntax_pass / total if total else 0.0,
        "pass_at_1": passed_tasks / len(with_tests) if with_tests else None,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else None,
        "passed_tasks": passed_tasks if with_tests else None,
        "total_tests": total_tests if with_tests else None,
        "passed_tests": passed_tests if with_tests else None,
        "exact_match": sum(1 for case in with_exact if case["exact_match"]) / len(with_exact) if with_exact else None,
        "avg_code_token_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "prompt": "CoT prompt with Reasoning, Key steps, and Final code sections.",
    }


def main() -> None:
    args = parse_args()
    raw_rows = load_rows(args.input_file)
    rows = [
        {
            "index": index,
            "task": row_to_task(row),
            "reference": normalize_text(row_to_reference(row)),
            "test_setup": row_to_tests(row)[0],
            "tests": row_to_tests(row)[1],
            "raw": row,
        }
        for index, row in enumerate(raw_rows)
    ]
    rows = [row for row in rows if row["task"]]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No valid tasks found in {args.input_file}")

    responses = generate_outputs(rows, args)
    cases = [score_case(row, response, args) for row, response in zip(rows, responses, strict=True)]
    metrics = summarize(cases, args)

    save_json(args.output_dir / "cot_code_metrics.json", metrics)
    save_jsonl(args.output_dir / "cot_code_cases.jsonl", cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to {args.output_dir / 'cot_code_metrics.json'}")
    print(f"Wrote cases to {args.output_dir / 'cot_code_cases.jsonl'}")


if __name__ == "__main__":
    main()
