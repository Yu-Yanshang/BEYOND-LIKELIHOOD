#!/usr/bin/env python3
"""Build a conservative preference subset for A2 Full SFT calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "best_path" / "runs" / "best_full_dpo_a5_20260702" / "data" / "a3_high_confidence.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dpo" / "data"


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_code_signal(text: str) -> bool:
    lowered = text.lower()
    return "```python" in lowered or "```py" in lowered or "\ndef " in text or text.lstrip().startswith("def ")


def length_ratio(a: str, b: str) -> float:
    small = max(1, min(len(a), len(b)))
    large = max(len(a), len(b))
    return large / small


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    instruction = normalize_text(row.get("instruction"))
    chosen = normalize_text(row.get("chosen"))
    rejected = normalize_text(row.get("rejected"))

    if not instruction:
        return False, "missing_instruction"
    if not chosen or not rejected:
        return False, "missing_response"
    if chosen == rejected:
        return False, "same_response"
    if len(chosen) < args.min_response_chars or len(rejected) < args.min_response_chars:
        return False, "response_too_short"
    if len(chosen) > args.max_response_chars or len(rejected) > args.max_response_chars:
        return False, "response_too_long"
    if length_ratio(chosen, rejected) > args.max_length_ratio:
        return False, "extreme_length_gap"
    if args.require_chosen_code and not has_code_signal(chosen):
        return False, "chosen_no_code_signal"
    return True, "kept"


def register_dataset(info_path: Path, train_name: str, eval_name: str) -> None:
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        info = {}

    columns = {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    info[train_name] = {
        "file_name": f"{train_name}.json",
        "ranking": True,
        "columns": columns,
    }
    info[eval_name] = {
        "file_name": f"{eval_name}.json",
        "ranking": True,
        "columns": columns,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-name", default="a2_pref_calib_train")
    parser.add_argument("--eval-name", default="a2_pref_calib_eval")
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--max-response-chars", type=int, default=6000)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--require-chosen-code", action="store_true", default=True)
    parser.add_argument("--allow-duplicate-prompts", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Missing preference source: {args.source}")

    rows = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {args.source}")

    counts: dict[str, int] = {}
    seen_prompts: set[str] = set()
    kept: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            counts["non_object"] = counts.get("non_object", 0) + 1
            continue
        ok, reason = keep_row(row, args)
        counts[reason] = counts.get(reason, 0) + 1
        if not ok:
            continue

        prompt_key = row_hash(normalize_text(row.get("instruction")) + "\n" + normalize_text(row.get("input")))
        if not args.allow_duplicate_prompts and prompt_key in seen_prompts:
            counts["duplicate_prompt"] = counts.get("duplicate_prompt", 0) + 1
            continue
        seen_prompts.add(prompt_key)

        kept.append(
            {
                "instruction": normalize_text(row.get("instruction")),
                "input": normalize_text(row.get("input")),
                "chosen": normalize_text(row.get("chosen")),
                "rejected": normalize_text(row.get("rejected")),
            }
        )

    needed = args.max_train + args.eval_size
    if len(kept) < needed:
        raise ValueError(f"Need at least {needed} kept rows, got {len(kept)}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    selected = kept[:needed]
    eval_rows = selected[: args.eval_size]
    train_rows = selected[args.eval_size :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.train_name}.json"
    eval_path = args.output_dir / f"{args.eval_name}.json"
    report_path = args.output_dir / f"{args.train_name}_report.json"

    train_path.write_text(json.dumps(train_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    eval_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    register_dataset(args.output_dir / "dataset_info.json", args.train_name, args.eval_name)

    report = {
        "source": str(args.source),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "train_name": args.train_name,
        "eval_name": args.eval_name,
        "seed": args.seed,
        "raw_rows": len(rows),
        "kept_rows": len(kept),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "filters": {
            "min_response_chars": args.min_response_chars,
            "max_response_chars": args.max_response_chars,
            "max_length_ratio": args.max_length_ratio,
            "require_chosen_code": args.require_chosen_code,
            "allow_duplicate_prompts": args.allow_duplicate_prompts,
        },
        "counts": counts,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
