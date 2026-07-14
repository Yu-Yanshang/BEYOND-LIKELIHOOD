#!/usr/bin/env python3
"""Build isolated configs, immutable inputs, and reports for the best-path run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path("/root/project")
RUNS_ROOT = PROJECT_ROOT / "best_path" / "runs"
BASE_MODEL = PROJECT_ROOT / "Qwen3-0.6B"
DEFAULT_A3_RUN = "a3_v2_verify_20260702_005837"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> int:
    data = read_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return len(data)


def dataset_entry(file_name: str, ranking: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input"},
    }
    if ranking:
        item["ranking"] = True
        item["columns"].update({"chosen": "chosen", "rejected": "rejected"})
    else:
        item["columns"]["response"] = "output"
    return item


def prepare(run_id: str, a3_run_id: str, a2_batch: int, a4_batch: int) -> None:
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"RUN_ID already exists: {run_dir}")
    for name in ("data", "configs", "models", "eval", "logs", "stages", "telemetry", "reports"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    sources = {
        "a1_train.json": PROJECT_ROOT / "sft/data/code_sft_train.json",
        "a1_valid.json": PROJECT_ROOT / "sft/data/code_sft_valid.json",
        "a1_test.json": PROJECT_ROOT / "sft/data/code_sft_test.json",
        "a3_high_confidence.json": PROJECT_ROOT / f"dpo/data/a3_v2_runs/{a3_run_id}/code_dpo_a3_v2_high_confidence_train.json",
    }
    source_manifest: dict[str, Any] = {}
    for name, source in sources.items():
        if not source.exists():
            raise FileNotFoundError(source)
        target = run_dir / "data" / name
        shutil.copy2(source, target)
        source_manifest[name] = {
            "source": str(source),
            "snapshot": str(target),
            "rows": rows(target),
            "sha256": sha256(target),
        }

    registry = {
        "best_a1_train": dataset_entry("a1_train.json"),
        "best_a1_valid": dataset_entry("a1_valid.json"),
        "best_a1_test": dataset_entry("a1_test.json"),
        "best_a3_high_confidence": dataset_entry("a3_high_confidence.json", ranking=True),
    }
    write_json(run_dir / "data/dataset_info.json", registry)

    a2_output = run_dir / "models/a2_full"
    a2_config = {
        "model_name_or_path": str(BASE_MODEL),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "dataset_dir": str(run_dir / "data"),
        "dataset": "best_a1_train",
        "eval_dataset": "best_a1_valid",
        "template": "qwen3",
        "enable_thinking": False,
        "cutoff_len": 2048,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 2,
        "output_dir": str(a2_output),
        "overwrite_output_dir": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "save_only_model": True,
        "eval_strategy": "epoch",
        "plot_loss": True,
        "report_to": "none",
        "skip_memory_metrics": False,
        "per_device_train_batch_size": a2_batch,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0e-5,
        "num_train_epochs": 2.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
        "seed": 42,
        "data_seed": 42,
    }
    (run_dir / "configs/a2_full.yaml").write_text(
        yaml.safe_dump(a2_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "schema_version": "best_path_v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "a3_run_id": a3_run_id,
        "base_model": str(BASE_MODEL),
        "contract": {
            "a2": "Full SFT, 2 epochs (prior formal run selected epoch 2), immutable current A1 snapshot",
            "a4": "LoRA-DPO from A2 Full, 2 epochs, A3 high-confidence data",
            "a5": "CoT + Best-of-8 + execution reranking + self-consistency tie-break + up to 2 Reflexion repairs",
            "template": "qwen3",
            "enable_thinking": False,
        },
        "batches": {"a2": a2_batch, "a4": a4_batch},
        "sources": source_manifest,
    }
    write_json(run_dir / "manifest.json", manifest)


def render_a4(run_id: str, a4_batch: int) -> None:
    run_dir = RUNS_ROOT / run_id
    a2_model = run_dir / "models/a2_full"
    if not (a2_model / "config.json").exists():
        raise FileNotFoundError(f"A2 model is incomplete: {a2_model}")
    config = {
        "model_name_or_path": str(a2_model),
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "finetuning_type": "lora",
        "pref_loss": "sigmoid",
        "pref_beta": 0.1,
        "ref_model": str(a2_model),
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "dataset_dir": str(run_dir / "data"),
        "dataset": "best_a3_high_confidence",
        "template": "qwen3",
        "enable_thinking": False,
        "cutoff_len": 1536,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 2,
        "output_dir": str(run_dir / "models/a4_dpo_adapter"),
        "overwrite_output_dir": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "save_only_model": True,
        "plot_loss": True,
        "report_to": "none",
        "skip_memory_metrics": False,
        "per_device_train_batch_size": a4_batch,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5.0e-5,
        "num_train_epochs": 2.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
        "seed": 42,
        "data_seed": 42,
        "eval_strategy": "no",
    }
    (run_dir / "configs/a4_lora_dpo.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def render_export(run_id: str) -> None:
    run_dir = RUNS_ROOT / run_id
    config = {
        "model_name_or_path": str(run_dir / "models/a2_full"),
        "adapter_name_or_path": str(run_dir / "models/a4_dpo_adapter"),
        "trust_remote_code": True,
        "finetuning_type": "lora",
        "template": "qwen3",
        "enable_thinking": False,
        "export_dir": str(run_dir / "models/a4_dpo_merged"),
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    (run_dir / "configs/a4_export.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def training_summary(path: Path) -> dict[str, Any]:
    result = path / "train_results.json"
    if result.exists():
        return read_json(result)
    return {}


def report(run_id: str) -> None:
    run_dir = RUNS_ROOT / run_id
    manifest = read_json(run_dir / "manifest.json")
    evals = {}
    for stage, rel in {
        "a2_full_greedy": "eval/a2_full/metrics.json",
        "a4_dpo_greedy": "eval/a4_dpo/metrics.json",
        "a5_enhanced": "eval/a5_enhanced/metrics.json",
    }.items():
        path = run_dir / rel
        evals[stage] = read_json(path) if path.exists() else {}
    value = {
        "schema_version": "best_path_report_v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "manifest": manifest,
        "training": {
            "a2_full": training_summary(run_dir / "models/a2_full"),
            "a4_lora_dpo": training_summary(run_dir / "models/a4_dpo_adapter"),
        },
        "evaluation": evals,
        "handoff": {
            "a2_full_model": str(run_dir / "models/a2_full"),
            "a4_adapter": str(run_dir / "models/a4_dpo_adapter"),
            "a4_merged_model": str(run_dir / "models/a4_dpo_merged"),
            "final_metrics": str(run_dir / "eval/a5_enhanced/metrics.json"),
        },
    }
    write_json(run_dir / "reports/final_report.json", value)
    lines = [
        f"# Best-path A2→A5 report: {run_id}", "",
        "| Stage | MBPP pass@1 | Syntax | Avg test pass | Passed tasks |", "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("A2 Full SFT (greedy)", "a2_full_greedy"), ("A4 LoRA-DPO (greedy)", "a4_dpo_greedy"), ("A5 enhanced", "a5_enhanced")):
        item = evals[key]
        lines.append(
            f"| {label} | {item.get('pass_at_1', '')} | {item.get('syntax_pass_rate', '')} | "
            f"{item.get('avg_test_pass_rate', '')} | {item.get('passed_tasks', '')} |"
        )
    lines += ["", f"Final merged model: `{value['handoff']['a4_merged_model']}`", "",
              "All rows use the same MBPP sanitized test split and the same execution harness."]
    (run_dir / "reports/final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--a3-run-id", default=DEFAULT_A3_RUN)
    prep.add_argument("--a2-batch", type=int, default=12)
    prep.add_argument("--a4-batch", type=int, default=6)
    a4 = sub.add_parser("render-a4")
    a4.add_argument("--run-id", required=True)
    a4.add_argument("--a4-batch", type=int, default=6)
    exp = sub.add_parser("render-export")
    exp.add_argument("--run-id", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_id, args.a3_run_id, args.a2_batch, args.a4_batch)
    elif args.command == "render-a4":
        render_a4(args.run_id, args.a4_batch)
    elif args.command == "render-export":
        render_export(args.run_id)
    elif args.command == "report":
        report(args.run_id)


if __name__ == "__main__":
    main()
