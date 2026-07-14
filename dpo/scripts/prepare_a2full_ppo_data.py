#!/usr/bin/env python3
"""Prepare PPO prompts for A2-full reward-function training.

The default dataset uses MBPP train+validation prompts with visible asserts.
Those asserts let the reward server execute lightweight tests while PPO is
sampling, which is a stronger signal than syntax-only rewards.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dpo" / "data"
DEFAULT_MBPP_DIR = Path("/root/mbpp")


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


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [normalize(item) for item in value if normalize(item)]
    text = normalize(value)
    return [text] if text else []


def format_mbpp_instruction(row: dict[str, Any]) -> str:
    prompt = normalize(row.get("prompt") or row.get("text"))
    setup = listify(row.get("test_imports") or row.get("test_setup_code"))
    tests = listify(row.get("test_list"))
    test_block = "\n".join([*setup, *tests])
    return normalize(
        f"""
You are an expert Python programmer. Return only self-contained Python code.

Task:
{prompt}

Your solution must pass these tests:
{test_block}

Output rules:
- Start immediately with the required def/class/import line after [BEGIN].
- Define the function or class names used by the tests.
- Do not include markdown fences, explanations, examples, sample print calls, or copied asserts.
- Do not output <think>, </think>, chat special tokens, or reasoning text.
- Stop after the complete solution; if you use [DONE], put nothing after it.

[BEGIN]
"""
    )


def build_mbpp_rows(mbpp_dir: Path, config: str, splits: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in splits:
        path = mbpp_dir / config / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing MBPP split: {path}")
        for item in read_parquet_rows(path):
            instruction = format_mbpp_instruction(item)
            if instruction:
                rows.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": "",
                    }
                )
    return rows


def build_dpo_prompt_rows(data_dir: Path, source: str, limit: int) -> list[dict[str, str]]:
    source_path = data_dir / source
    if not source_path.exists() or limit == 0:
        return []
    raw_rows = read_json(source_path)
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected list in {source_path}")

    output = []
    for item in raw_rows:
        instruction = normalize(item.get("instruction"))
        input_text = normalize(item.get("input"))
        if instruction:
            output.append({"instruction": instruction, "input": input_text, "output": ""})
        if limit > 0 and len(output) >= limit:
            break
    return output


def deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    output = []
    for row in rows:
        key = (row["instruction"], row.get("input", ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def update_dataset_info(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    if dataset_info_path.exists():
        info = read_json(dataset_info_path)
        if not isinstance(info, dict):
            raise ValueError(f"Expected object in {dataset_info_path}")
    else:
        info = {}

    info[dataset_name] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }
    write_json(dataset_info_path, info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mbpp-dir", type=Path, default=DEFAULT_MBPP_DIR)
    parser.add_argument("--mbpp-config", default="sanitized")
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--output", default="a2full_reward_ppo_train.json")
    parser.add_argument("--dataset-name", default="a2full_reward_ppo_train")
    parser.add_argument("--limit", type=int, default=0, help="0 keeps all prepared rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpo-source", default="code_dpo_chain_train.json")
    parser.add_argument("--dpo-limit", type=int, default=0, help="Optionally mix in prompt-only DPO rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    rows = build_mbpp_rows(args.mbpp_dir, args.mbpp_config, splits)
    dpo_rows = build_dpo_prompt_rows(args.data_dir, args.dpo_source, args.dpo_limit)
    rows = deduplicate_rows([*rows, *dpo_rows])
    random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No PPO rows were produced.")

    output_path = args.data_dir / args.output
    dataset_info_path = args.data_dir / "dataset_info.json"
    report_path = args.data_dir / f"{Path(args.output).stem}_report.json"
    write_json(output_path, rows)
    update_dataset_info(dataset_info_path, args.dataset_name, args.output)
    write_json(
        report_path,
        {
            "dataset_name": args.dataset_name,
            "output": str(output_path),
            "rows": len(rows),
            "mbpp_dir": str(args.mbpp_dir),
            "mbpp_config": args.mbpp_config,
            "splits": splits,
            "dpo_source": args.dpo_source,
            "dpo_rows_requested": args.dpo_limit,
            "seed": args.seed,
        },
    )

    print(f"Wrote PPO rows: {len(rows)} -> {output_path}")
    print(f"Registered dataset: {args.dataset_name} in {dataset_info_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
