#!/usr/bin/env python3
"""Run and evaluate base-vs-PPO generation on Python code tasks.

This script does not execute generated code. It checks syntax with ast.parse and
compares generated code to the chosen response using exact match and token F1.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMA_FACTORY_DIR = PROJECT_ROOT / "LlamaFactory"
DEFAULT_BASE_CONFIG = Path("dpo/configs/qwen3_06b_code_predict_base.yaml")
DEFAULT_PPO_CONFIG = Path("dpo/configs/qwen3_06b_code_predict_ppo.yaml")
DEFAULT_BASE_PREDICTIONS = PROJECT_ROOT / "dpo" / "outputs" / "predict_base_on_code_dpo" / "generated_predictions.jsonl"
DEFAULT_PPO_PREDICTIONS = PROJECT_ROOT / "dpo" / "outputs" / "predict_ppo_on_code_dpo" / "generated_predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dpo" / "outputs" / "code_test_ppo"

FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def extract_code(text: str) -> str:
    matches = FENCED_CODE_RE.findall(str(text))
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


def token_f1(prediction: str, reference: str) -> float:
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


def iter_prediction_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def score_prediction_item(line_no: int, item: dict) -> dict:
    prediction = normalize_text(item.get("predict", ""))
    reference = normalize_text(item.get("label", ""))
    pred_code = extract_code(prediction)
    ref_code = extract_code(reference)
    f1 = token_f1(pred_code, ref_code)

    return {
        "line_no": line_no,
        "prompt": item.get("prompt", ""),
        "predict": prediction,
        "label": reference,
        "code": pred_code,
        "syntax_ok": syntax_ok(pred_code),
        "exact_match": pred_code == ref_code,
        "token_f1": f1,
    }


def score_predictions(path: Path) -> tuple[dict, list[dict]]:
    total = 0
    exact = 0
    syntax_pass = 0
    f1_sum = 0.0
    cases = []

    for line_no, item in iter_prediction_rows(path):
        case = score_prediction_item(line_no, item)

        total += 1
        exact += int(case["exact_match"])
        syntax_pass += int(case["syntax_ok"])
        f1_sum += case["token_f1"]
        cases.append(case)

    return (
        {
            "predictions": str(path),
            "total": total,
            "exact_match": exact / total if total else 0.0,
            "syntax_pass_rate": syntax_pass / total if total else 0.0,
            "avg_code_token_f1": f1_sum / total if total else 0.0,
        },
        cases,
    )


def build_compare_cases(base_cases: list[dict], ppo_cases: list[dict], limit: int) -> list[dict]:
    compare_cases = []
    for base_case, ppo_case in zip(base_cases, ppo_cases, strict=True):
        compare_cases.append(
            {
                "line_no": base_case["line_no"],
                "prompt": base_case["prompt"],
                "label": base_case["label"],
                "base_predict": base_case["predict"],
                "ppo_predict": ppo_case["predict"],
                "base_syntax_ok": base_case["syntax_ok"],
                "ppo_syntax_ok": ppo_case["syntax_ok"],
                "base_exact_match": base_case["exact_match"],
                "ppo_exact_match": ppo_case["exact_match"],
                "base_token_f1": base_case["token_f1"],
                "ppo_token_f1": ppo_case["token_f1"],
            }
        )
        if len(compare_cases) >= limit:
            break
    return compare_cases


def run_predict(config: Path, python_bin: str, gpu_id: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", gpu_id)
    env["PYTHONPATH"] = f"{LLAMA_FACTORY_DIR / 'src'}:{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"Running prediction config: {config}")
    subprocess.run([python_bin, "-m", "llamafactory.cli", "train", str(config)], cwd=PROJECT_ROOT, env=env, check=True)


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--gpu_id", default="3")
    parser.add_argument("--base_config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--ppo_config", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--base_predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--ppo_predictions", type=Path, default=DEFAULT_PPO_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case_limit", type=int, default=200)
    parser.add_argument("--skip_predict", action="store_true", help="Only evaluate existing generated_predictions.jsonl files.")
    args = parser.parse_args()

    if not args.skip_predict:
        run_predict(args.base_config, args.python_bin, args.gpu_id)
        run_predict(args.ppo_config, args.python_bin, args.gpu_id)

    if not args.base_predictions.exists():
        raise FileNotFoundError(f"Missing base predictions: {args.base_predictions}")
    if not args.ppo_predictions.exists():
        raise FileNotFoundError(f"Missing PPO predictions: {args.ppo_predictions}")

    base_metrics, base_cases = score_predictions(args.base_predictions)
    ppo_metrics, ppo_cases = score_predictions(args.ppo_predictions)
    if base_metrics["total"] != ppo_metrics["total"]:
        raise ValueError(f"Base rows ({base_metrics['total']}) != PPO rows ({ppo_metrics['total']})")

    comparison = {
        "base": base_metrics,
        "ppo": ppo_metrics,
        "delta_ppo_minus_base": {
            "exact_match": ppo_metrics["exact_match"] - base_metrics["exact_match"],
            "syntax_pass_rate": ppo_metrics["syntax_pass_rate"] - base_metrics["syntax_pass_rate"],
            "avg_code_token_f1": ppo_metrics["avg_code_token_f1"] - base_metrics["avg_code_token_f1"],
        },
        "note": "Generated code is parsed but never executed.",
    }

    save_json(args.output_dir / "code_ppo_test_metrics.json", comparison)
    save_jsonl(args.output_dir / "code_ppo_compare_cases.jsonl", build_compare_cases(base_cases, ppo_cases, args.case_limit))

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to {args.output_dir / 'code_ppo_test_metrics.json'}")
    print(f"Wrote compare cases to {args.output_dir / 'code_ppo_compare_cases.jsonl'}")


if __name__ == "__main__":
    main()
