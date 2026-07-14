#!/usr/bin/env python3
"""Compare A2-full baseline and PPO MBPP metrics with anti-regression gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def metric(data: dict[str, Any], name: str) -> float:
    value = data.get(name, 0.0)
    return float(value if value is not None else 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-metrics", type=Path, required=True)
    parser.add_argument("--ppo-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pass-drop", type=float, default=0.02)
    parser.add_argument("--max-test-drop", type=float, default=0.03)
    parser.add_argument("--max-syntax-drop", type=float, default=0.05)
    parser.add_argument("--require-any-gain", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = read_json(args.base_metrics)
    ppo = read_json(args.ppo_metrics)

    deltas = {
        "pass_at_1": metric(ppo, "pass_at_1") - metric(base, "pass_at_1"),
        "avg_test_pass_rate": metric(ppo, "avg_test_pass_rate") - metric(base, "avg_test_pass_rate"),
        "syntax_pass_rate": metric(ppo, "syntax_pass_rate") - metric(base, "syntax_pass_rate"),
    }
    checks = {
        "pass_at_1_not_degraded": deltas["pass_at_1"] >= -args.max_pass_drop,
        "avg_test_pass_rate_not_degraded": deltas["avg_test_pass_rate"] >= -args.max_test_drop,
        "syntax_not_degraded": deltas["syntax_pass_rate"] >= -args.max_syntax_drop,
    }
    if args.require_any_gain:
        checks["any_metric_improved"] = any(value > 0 for value in deltas.values())

    report = {
        "base_metrics": str(args.base_metrics),
        "ppo_metrics": str(args.ppo_metrics),
        "thresholds": {
            "max_pass_drop": args.max_pass_drop,
            "max_test_drop": args.max_test_drop,
            "max_syntax_drop": args.max_syntax_drop,
            "require_any_gain": args.require_any_gain,
        },
        "base": {
            "pass_at_1": metric(base, "pass_at_1"),
            "avg_test_pass_rate": metric(base, "avg_test_pass_rate"),
            "syntax_pass_rate": metric(base, "syntax_pass_rate"),
        },
        "ppo": {
            "pass_at_1": metric(ppo, "pass_at_1"),
            "avg_test_pass_rate": metric(ppo, "avg_test_pass_rate"),
            "syntax_pass_rate": metric(ppo, "syntax_pass_rate"),
        },
        "delta_ppo_minus_base": deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
