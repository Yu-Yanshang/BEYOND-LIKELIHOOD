#!/usr/bin/env python3
"""Batch before-vs-after inference for Python code tasks.

The script loads one input file, runs the same CoT prompt through two models,
and writes side-by-side outputs plus lightweight code metrics. It never executes
generated code.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE_MODEL = PROJECT_ROOT / "Qwen3-0.6B"
DEFAULT_AFTER_MODEL = PROJECT_ROOT / "dpo" / "outputs" / "qwen3_06b_code_full_dpo"
DEFAULT_DPO_TEST = PROJECT_ROOT / "dpo" / "data" / "code_dpo_test.json"
DEFAULT_SFT_TEST = PROJECT_ROOT / "sft" / "data" / "code_sft_test.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tts" / "outputs" / "batch_compare"

FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")

try:
    from cot_code_example import build_chat_messages
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from cot_code_example import build_chat_messages


def normalize_text(text: Any) -> str:
    lines = [line.rstrip() for line in str(text or "").strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def extract_code(text: Any) -> str:
    matches = FENCED_CODE_RE.findall(str(text or ""))
    if matches:
        return normalize_text(matches[0])
    return normalize_text(text)


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

    prompt = first_text(row, ("prompt", "instruction", "question", "task", "task_description"))
    input_text = first_text(row, ("input", "query"))
    if prompt and input_text:
        return f"{prompt}\n\n{input_text}"
    return prompt or input_text


def row_to_reference(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return first_text(row, ("output", "chosen", "answer", "reference", "label"))


def load_rows(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]

    return [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]


def default_input_file() -> Path:
    if DEFAULT_DPO_TEST.exists():
        return DEFAULT_DPO_TEST
    return DEFAULT_SFT_TEST


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


def generate_batch(model: Any, tokenizer: Any, tasks: list[str], args: argparse.Namespace) -> list[str]:
    import torch

    prompts = [format_chat(tokenizer, task) for task in tasks]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_tokens).to(
        model_input_device(model)
    )
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
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)


def run_model(model_path: Path, rows: list[dict[str, Any]], args: argparse.Namespace, label: str) -> list[str]:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {label} model directory: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {label} model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    outputs: list[str] = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_tasks = [row["task"] for row in batch_rows]
        outputs.extend(normalize_text(text) for text in generate_batch(model, tokenizer, batch_tasks, args))
        print(f"{label}: generated {min(start + args.batch_size, len(rows))}/{len(rows)}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


def score_response(response: str, reference: str) -> dict[str, Any]:
    pred_code = extract_code(response)
    ref_code = extract_code(reference)
    return {
        "code": pred_code,
        "syntax_ok": syntax_ok(pred_code),
        "exact_match": pred_code == ref_code if ref_code else None,
        "token_f1": token_f1(pred_code, ref_code),
    }


def summarize(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(rows)
    with_reference = [row for row in rows if row["reference_code"]]
    exact_values = [row[f"{prefix}_exact_match"] for row in rows if row[f"{prefix}_exact_match"] is not None]
    f1_values = [row[f"{prefix}_token_f1"] for row in rows if row[f"{prefix}_token_f1"] is not None]
    syntax_ok_count = sum(1 for row in rows if row[f"{prefix}_syntax_ok"])

    return {
        "total": total,
        "with_reference": len(with_reference),
        "syntax_pass_rate": syntax_ok_count / total if total else 0.0,
        "exact_match": sum(1 for value in exact_values if value) / len(exact_values) if exact_values else None,
        "avg_code_token_f1": sum(f1_values) / len(f1_values) if f1_values else None,
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


def save_markdown(path: Path, rows: list[dict[str, Any]], limit: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Batch Inference Comparison", ""]
    for row in rows[:limit]:
        lines.extend(
            [
                f"## Case {row['index']}",
                "",
                "### Task",
                "",
                row["task"],
                "",
            ]
        )
        if row["reference"]:
            lines.extend(["### Reference", "", row["reference"], ""])
        lines.extend(
            [
                "### Before",
                "",
                row["before_response"],
                "",
                "### After",
                "",
                row["after_response"],
                "",
                (
                    f"`before_syntax_ok={row['before_syntax_ok']}` "
                    f"`after_syntax_ok={row['after_syntax_ok']}` "
                    f"`before_f1={row['before_token_f1']}` "
                    f"`after_f1={row['after_token_f1']}`"
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, default=None, help="json/jsonl/txt tasks. Default: dpo test, else sft test.")
    parser.add_argument("--before_model", type=Path, default=DEFAULT_BEFORE_MODEL)
    parser.add_argument("--after_model", type=Path, default=DEFAULT_AFTER_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20, help="0 means use all rows.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--markdown_cases", type=int, default=20)
    args = parser.parse_args()

    input_file = args.input_file or default_input_file()
    if not args.before_model.exists():
        raise FileNotFoundError(f"Missing before model directory: {args.before_model}")
    if not args.after_model.exists():
        raise FileNotFoundError(
            f"Missing after model directory: {args.after_model}. Train it first with: bash dpo/scripts/train.sh"
        )

    raw_rows = load_rows(input_file)
    normalized_rows = [
        {
            "index": index,
            "task": row_to_task(row),
            "reference": normalize_text(row_to_reference(row)),
            "raw": row,
        }
        for index, row in enumerate(raw_rows)
    ]
    normalized_rows = [row for row in normalized_rows if row["task"]]
    if args.limit > 0:
        normalized_rows = normalized_rows[: args.limit]
    if not normalized_rows:
        raise RuntimeError(f"No valid tasks found in {input_file}")

    before_outputs = run_model(args.before_model, normalized_rows, args, "before")
    after_outputs = run_model(args.after_model, normalized_rows, args, "after")

    comparison_rows = []
    for row, before_response, after_response in zip(normalized_rows, before_outputs, after_outputs, strict=True):
        before_score = score_response(before_response, row["reference"])
        after_score = score_response(after_response, row["reference"])
        reference_code = extract_code(row["reference"])
        comparison_rows.append(
            {
                "index": row["index"],
                "task": row["task"],
                "reference": row["reference"],
                "reference_code": reference_code,
                "before_response": before_response,
                "after_response": after_response,
                "before_code": before_score["code"],
                "after_code": after_score["code"],
                "before_syntax_ok": before_score["syntax_ok"],
                "after_syntax_ok": after_score["syntax_ok"],
                "before_exact_match": before_score["exact_match"],
                "after_exact_match": after_score["exact_match"],
                "before_token_f1": before_score["token_f1"],
                "after_token_f1": after_score["token_f1"],
            }
        )

    before_metrics = summarize(comparison_rows, "before")
    after_metrics = summarize(comparison_rows, "after")
    metrics = {
        "input_file": str(input_file),
        "before_model": str(args.before_model),
        "after_model": str(args.after_model),
        "before": before_metrics,
        "after": after_metrics,
        "delta_after_minus_before": {
            "syntax_pass_rate": after_metrics["syntax_pass_rate"] - before_metrics["syntax_pass_rate"],
            "exact_match": (
                after_metrics["exact_match"] - before_metrics["exact_match"]
                if after_metrics["exact_match"] is not None and before_metrics["exact_match"] is not None
                else None
            ),
            "avg_code_token_f1": (
                after_metrics["avg_code_token_f1"] - before_metrics["avg_code_token_f1"]
                if after_metrics["avg_code_token_f1"] is not None and before_metrics["avg_code_token_f1"] is not None
                else None
            ),
        },
        "note": "Generated Python is parsed with ast.parse but never executed.",
    }

    save_json(args.output_dir / "metrics.json", metrics)
    save_jsonl(args.output_dir / "comparison.jsonl", comparison_rows)
    save_markdown(args.output_dir / "comparison.md", comparison_rows, args.markdown_cases)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {args.output_dir / 'metrics.json'}")
    print(f"Wrote side-by-side JSONL: {args.output_dir / 'comparison.jsonl'}")
    print(f"Wrote readable report: {args.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
