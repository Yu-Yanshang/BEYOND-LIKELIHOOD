#!/usr/bin/env python3
"""Compare Base, DPO and PPO MBPP metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_KEYS = [
    "pass_at_1",
    "syntax_pass_rate",
    "avg_test_pass_rate",
    "passed_tasks",
    "passed_tests",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def delta(left: dict[str, Any], right: dict[str, Any], key: str) -> float | int:
    if key in {"passed_tasks", "passed_tests"}:
        return int(left.get(key, 0)) - int(right.get(key, 0))
    return float(left.get(key, 0.0)) - float(right.get(key, 0.0))


def build_delta(left_name: str, left: dict[str, Any], right_name: str, right: dict[str, Any]) -> dict[str, Any]:
    return {key: delta(left, right, key) for key in METRIC_KEYS}


def select_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_metrics", type=Path, required=True)
    parser.add_argument("--dpo_metrics", type=Path, required=True)
    parser.add_argument("--ppo_metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = read_json(args.base_metrics)
    dpo = read_json(args.dpo_metrics)
    ppo = read_json(args.ppo_metrics)

    result = {
        "base": select_metrics(base),
        "dpo": select_metrics(dpo),
        "ppo": select_metrics(ppo),
        "delta_dpo_minus_base": build_delta("dpo", dpo, "base", base),
        "delta_ppo_minus_base": build_delta("ppo", ppo, "base", base),
        "delta_ppo_minus_dpo": build_delta("ppo", ppo, "dpo", dpo),
        "paths": {
            "base_metrics": str(args.base_metrics),
            "dpo_metrics": str(args.dpo_metrics),
            "ppo_metrics": str(args.ppo_metrics),
        },
    }
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
