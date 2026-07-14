#!/usr/bin/env python3
"""Build prompt-only PPO data from the existing DPO ranking dataset.

The existing project uses LLaMA-Factory ranking data:
  {"instruction": ..., "input": ..., "chosen": ..., "rejected": ...}

Reward-model training can reuse that ranking dataset directly. PPO training only
needs prompts; LLaMA-Factory still expects a normal instruction-style dataset
entry, so this script writes an empty output field and registers it in
`dataset_info.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dpo" / "data"


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


def build_ppo_rows(rows: list[dict[str, Any]], deduplicate: bool) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, str]] = []

    for item in rows:
        instruction = normalize(item.get("instruction"))
        input_text = normalize(item.get("input"))
        if not instruction:
            continue

        key = (instruction, input_text)
        if deduplicate and key in seen:
            continue
        seen.add(key)

        output.append(
            {
                "instruction": instruction,
                "input": input_text,
                # PPO ignores supervised labels, but the standard alpaca-style
                # loader is most robust when the response column exists.
                "output": "",
            }
        )

    return output


def update_dataset_info(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    if dataset_info_path.exists():
        info = read_json(dataset_info_path)
        if not isinstance(info, dict):
            raise ValueError(f"dataset_info.json must contain an object: {dataset_info_path}")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source", default="code_dpo_train.json")
    parser.add_argument("--output", default="code_ppo_train.json")
    parser.add_argument("--dataset_name", default="code_ppo_train")
    parser.add_argument("--no_deduplicate", action="store_true")
    args = parser.parse_args()

    source_path = args.data_dir / args.source
    output_path = args.data_dir / args.output
    dataset_info_path = args.data_dir / "dataset_info.json"

    rows = read_json(source_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {source_path}")

    ppo_rows = build_ppo_rows(rows, deduplicate=not args.no_deduplicate)
    if not ppo_rows:
        raise RuntimeError(f"No PPO rows were produced from {source_path}")

    write_json(output_path, ppo_rows)
    update_dataset_info(dataset_info_path, args.dataset_name, args.output)

    print(f"Read DPO rows   : {len(rows)}")
    print(f"Wrote PPO rows  : {len(ppo_rows)}")
    print(f"PPO data        : {output_path}")
    print(f"Dataset registry: {dataset_info_path} -> {args.dataset_name}")


if __name__ == "__main__":
    main()
