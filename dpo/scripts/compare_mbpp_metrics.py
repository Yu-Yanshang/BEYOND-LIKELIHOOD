#!/usr/bin/env python3
"""Compare base and DPO MBPP metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def metric_delta(dpo: dict[str, Any], base: dict[str, Any], key: str) -> float:
    return float(dpo.get(key, 0.0)) - float(base.get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_metrics", type=Path, required=True)
    parser.add_argument("--dpo_metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = read_json(args.base_metrics)
    dpo = read_json(args.dpo_metrics)
    comparison = {
        "base": base,
        "dpo": dpo,
        "delta_dpo_minus_base": {
            "pass_at_1": metric_delta(dpo, base, "pass_at_1"),
            "syntax_pass_rate": metric_delta(dpo, base, "syntax_pass_rate"),
            "avg_test_pass_rate": metric_delta(dpo, base, "avg_test_pass_rate"),
            "passed_tasks": int(dpo.get("passed_tasks", 0)) - int(base.get("passed_tasks", 0)),
            "passed_tests": int(dpo.get("passed_tests", 0)) - int(base.get("passed_tests", 0)),
        },
    }
    save_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
