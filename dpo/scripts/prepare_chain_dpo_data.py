#!/usr/bin/env python3
"""Create chain-specific DPO train/eval files without mutating source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dpo/data"))
    parser.add_argument("--source", default="code_dpo_train.json")
    parser.add_argument("--train-name", default="code_dpo_chain_train")
    parser.add_argument("--eval-name", default="code_dpo_chain_eval")
    parser.add_argument("--eval-size", type=int, default=512)
    args = parser.parse_args()

    source_path = args.data_dir / args.source
    train_path = args.data_dir / f"{args.train_name}.json"
    eval_path = args.data_dir / f"{args.eval_name}.json"
    info_path = args.data_dir / "dataset_info.json"

    rows = json.loads(source_path.read_text(encoding="utf-8"))
    if len(rows) <= args.eval_size:
        raise ValueError(f"Need more than {args.eval_size} rows in {source_path}")

    if not train_path.exists():
        train_path.write_text(json.dumps(rows[:-args.eval_size], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not eval_path.exists():
        eval_path.write_text(json.dumps(rows[-args.eval_size :], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    ranking_columns = {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    info[args.train_name] = {
        "file_name": f"{args.train_name}.json",
        "ranking": True,
        "columns": ranking_columns,
    }
    info[args.eval_name] = {
        "file_name": f"{args.eval_name}.json",
        "ranking": True,
        "columns": ranking_columns,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train={train_path} rows={len(rows) - args.eval_size}")
    print(f"eval={eval_path} rows={args.eval_size}")
    print(f"updated={info_path}")


if __name__ == "__main__":
    main()
