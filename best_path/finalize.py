#!/usr/bin/env python3
"""Aggregate the two A5 routes and all upstream checkpoints into one report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    metrics = {
        "a2_full_greedy": load(run / "eval/a2_full/metrics.json"),
        "a4_dpo_epoch1_greedy": load(run / "eval/a4_dpo_epoch1/metrics.json"),
        "a4_dpo_epoch2_greedy": load(run / "eval/a4_dpo/metrics.json"),
        "a4_dpo_epoch2_legacy_protocol": load(run / "eval/a4_dpo_legacy_protocol/mbpp_metrics.json"),
        "a5_from_a2_full": load(run / "eval/a5_from_a2_full/metrics.json"),
        "a5_from_a4_dpo": load(run / "eval/a5_from_a4_dpo/metrics.json"),
    }
    report = {
        "schema_version": "best_path_dual_a5_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run.name,
        "manifest": load(run / "manifest.json"),
        "selection": {
            "a4_checkpoint": "epoch2",
            "reason": "Epoch 1 and 2 tied on pass@1; epoch 2 had higher syntax pass rate.",
            "final_quality_gate": "Both Full-SFT and A4-DPO are evaluated independently through A5.",
        },
        "metrics": metrics,
        "training": {
            "a2_full": load(run / "models/a2_full/train_results.json"),
            "a4_lora_dpo": load(run / "models/a4_dpo_adapter/train_results.json"),
        },
        "artifacts": {
            "a2_full_model": str(run / "models/a2_full"),
            "a4_epoch2_merged_model": str(run / "models/a4_dpo_merged"),
            "a5_full_cases": str(run / "eval/a5_from_a2_full/cases.jsonl"),
            "a5_dpo_cases": str(run / "eval/a5_from_a4_dpo/cases.jsonl"),
        },
    }
    write(run / "reports/final_report.json", report)

    rows = [
        ("A2 Full SFT, greedy", metrics["a2_full_greedy"]),
        ("A4 DPO epoch 1, greedy", metrics["a4_dpo_epoch1_greedy"]),
        ("A4 DPO epoch 2, greedy", metrics["a4_dpo_epoch2_greedy"]),
        ("A4 DPO epoch 2, legacy prompt", metrics["a4_dpo_epoch2_legacy_protocol"]),
        ("A5 from Full SFT", metrics["a5_from_a2_full"]),
        ("A5 from A4 DPO", metrics["a5_from_a4_dpo"]),
    ]
    lines = [
        f"# A2→A5 dual-route report: {run.name}", "",
        "| Route | pass@1 | Syntax pass | Avg test pass | Passed tasks | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, item in rows:
        lines.append(
            f"| {label} | {item.get('pass_at_1', '')} | {item.get('syntax_pass_rate', '')} | "
            f"{item.get('avg_test_pass_rate', '')} | {item.get('passed_tasks', '')} | "
            f"{item.get('runtime_seconds', '')} |"
        )
    lines += [
        "", "## Selection notes", "",
        "- A4 epoch 2 is used for the DPO→A5 route because epoch 1 and epoch 2 tie on pass@1 while epoch 2 has higher syntax pass rate.",
        "- Full-SFT→A5 and DPO→A5 are both retained; the final best score is selected only after both complete.",
        "- `enable_thinking` is false for SFT, DPO, merging, and both A5 routes.",
        "- The legacy-prompt row is the direct comparison point for the earlier 19.84% LoRA-DPO result.",
    ]
    (run / "reports/final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
