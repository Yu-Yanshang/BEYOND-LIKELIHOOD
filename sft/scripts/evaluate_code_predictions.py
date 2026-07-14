#!/usr/bin/env python3
"""Evaluate SFT predictions on the MBPP sanitized test subset with actual execution checks.

This script reads a generated_predictions.jsonl file from the SFT pipeline and
runs each candidate solution against the MBPP test cases. The score reported is
pass@1 (fraction of tasks where all provided tests pass), together with syntax
and average test-pass rates.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MBPP_DIR = PROJECT_ROOT.parent / "mbpp"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "sft" / "outputs" / "qwen3_06b_code_full_predict" / "generated_predictions.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sft" / "outputs" / "eval_mbpp"

FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


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


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except Exception:
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict("records")
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet file: {path}") from exc


def load_split(mbpp_dir: Path, config: str, split: str) -> list[dict[str, Any]]:
    path = mbpp_dir / config / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing MBPP parquet split: {path}")
    return read_parquet_rows(path)


def row_prompt(row: dict[str, Any], config: str) -> str:
    key = "prompt" if config == "sanitized" else "text"
    return normalize_text(row.get(key))


def row_setup(row: dict[str, Any], config: str) -> str:
    if config == "sanitized":
        return "\n".join(listify(row.get("test_imports")))
    return normalize_text(row.get("test_setup_code"))


def row_tests(row: dict[str, Any], include_challenge: bool) -> list[str]:
    tests = listify(row.get("test_list"))
    if include_challenge:
        tests.extend(listify(row.get("challenge_test_list")))
    return tests


def extract_code(text: str) -> str:
    text = normalize_text(text)
    if "[BEGIN]" in text:
        text = text.rsplit("[BEGIN]", 1)[-1]
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]

    matches = FENCED_CODE_RE.findall(text)
    if matches:
        return normalize_text(matches[-1])

    fence_positions = [text.lower().rfind(marker) for marker in ("```python", "```py", "```")]
    fence_pos = max(fence_positions)
    if fence_pos >= 0:
        candidate = text[fence_pos:].split("\n", 1)
        if len(candidate) == 2:
            text = candidate[1]
            if "```" in text:
                text = text.split("```", 1)[0]
            return normalize_text(text)

    raw_lines = text.splitlines()
    first_code = 0
    for idx, line in enumerate(raw_lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
            first_code = idx
            break

    lines = []
    for line in raw_lines[first_code:]:
        stripped = line.strip()
        if stripped in {"[DONE]", "DONE"}:
            break
        if stripped.startswith("```"):
            continue
        lines.append(line)
    return normalize_text("\n".join(lines))


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


def run_one_assert(code: str, setup_code: str, test: str, timeout: float, memory_mb: int) -> dict[str, Any]:
    runner = "\n\n".join(
        part for part in [
            "import faulthandler\nfaulthandler.enable()",
            setup_code,
            code,
            test,
        ]
        if normalize_text(part)
    )

    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as tmpdir:
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
                timeout=timeout,
                preexec_fn=lambda: limit_resources(memory_mb, timeout) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error_type": "timeout", "stderr": str(exc)}

    return {
        "passed": result.returncode == 0,
        "error_type": "" if result.returncode == 0 else "runtime_error",
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-2000:],
    }


def iter_prediction_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            yield line_no, item


def choose_prediction_text(item: dict[str, Any]) -> str:
    for key in ("predict", "response", "generated_text", "output", "completion", "prediction"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mbpp_dir", type=Path, default=DEFAULT_MBPP_DIR)
    parser.add_argument("--config", choices=("sanitized", "full"), default="sanitized")
    parser.add_argument("--split", default="test")
    parser.add_argument("--case_limit", type=int, default=200, help="Number of detailed cases to save.")
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all rows.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--test_timeout", type=float, default=5.0)
    parser.add_argument("--memory_mb", type=int, default=1024)
    parser.add_argument("--include_challenge_tests", action="store_true")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Missing predictions file: {args.predictions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = load_split(args.mbpp_dir, args.config, args.split)
    eval_rows = eval_rows[args.start_index :]
    if args.limit > 0:
        eval_rows = eval_rows[: args.limit]

    task_id_to_row = {int(row.get("task_id")): row for row in eval_rows if str(row.get("task_id")).strip()}

    prediction_rows = list(iter_prediction_rows(args.predictions))
    if args.limit > 0:
        prediction_rows = prediction_rows[: args.limit]

    if len(prediction_rows) != len(eval_rows):
        print(
            f"Warning: prediction rows ({len(prediction_rows)}) and MBPP rows ({len(eval_rows)}) differ. "
            "The script will match by task_id when possible and otherwise fall back to line order.",
            flush=True,
        )

    detailed_cases = []
    total = 0
    syntax_pass = 0
    passed = 0
    passed_tests = 0
    total_tests = 0
    f1_sum = 0.0
    exact = 0

    for idx, (line_no, item) in enumerate(prediction_rows):
        prediction_text = choose_prediction_text(item)
        pred_code = extract_code(prediction_text)
        is_syntax_ok = syntax_ok(pred_code)
        syntax_pass += int(is_syntax_ok)

        reference_row = None
        raw_task_id = item.get("task_id")
        if raw_task_id is not None and int(raw_task_id) in task_id_to_row:
            reference_row = task_id_to_row[int(raw_task_id)]
        elif idx < len(eval_rows):
            reference_row = eval_rows[idx]

        if reference_row is None:
            continue

        code = normalize_text(pred_code)
        tests = row_tests(reference_row, args.include_challenge_tests)
        setup_code = row_setup(reference_row, args.config)
        per_test = [run_one_assert(code, setup_code, test, args.test_timeout, args.memory_mb) for test in tests]
        test_passed = sum(1 for item in per_test if item["passed"])
        total_tests += len(tests)
        passed_tests += test_passed

        is_exact = code == normalize_text(reference_row.get("code", ""))
        exact += int(is_exact)
        f1_sum += token_f1(code, normalize_text(reference_row.get("code", "")))

        total += 1
        passed += int(test_passed == len(tests) and len(tests) > 0)

        if len(detailed_cases) < args.case_limit:
            detailed_cases.append(
                {
                    "line_no": line_no,
                    "task_id": int(reference_row.get("task_id", idx)),
                    "prompt": row_prompt(reference_row, args.config),
                    "predict": prediction_text,
                    "reference_code": normalize_text(reference_row.get("code", "")),
                    "code": code,
                    "syntax_ok": is_syntax_ok,
                    "passed": test_passed == len(tests) and len(tests) > 0,
                    "passed_tests": test_passed,
                    "total_tests": len(tests),
                    "test_results": per_test,
                    "exact_match": is_exact,
                    "token_f1": token_f1(code, normalize_text(reference_row.get("code", ""))),
                }
            )

    metrics = {
        "benchmark": "MBPP",
        "config": args.config,
        "split": args.split,
        "predictions": str(args.predictions),
        "mbpp_dir": str(args.mbpp_dir),
        "num_tasks": total,
        "pass_at_1": passed / total if total else 0.0,
        "syntax_pass_rate": syntax_pass / total if total else 0.0,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "exact_match": exact / total if total else 0.0,
        "avg_code_token_f1": f1_sum / total if total else 0.0,
        "passed_tasks": passed,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "note": "Reports pass@1 using MBPP sanitized task tests and execution outputs.",
    }

    metrics_path = args.output_dir / "mbpp_metrics.json"
    cases_path = args.output_dir / "mbpp_cases.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with cases_path.open("w", encoding="utf-8") as f:
        for case in detailed_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote detailed cases to {cases_path}")


if __name__ == "__main__":
    main()
