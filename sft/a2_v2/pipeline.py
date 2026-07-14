#!/usr/bin/env python3
"""Prepare, render, select, and report isolated A2 v2 experiments."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_ROOT = PROJECT_ROOT / "sft"
DATA_RUN_ROOT = SFT_ROOT / "data" / "a2_v2_runs"
OUTPUT_RUN_ROOT = SFT_ROOT / "outputs" / "a2_v2_runs"
BASE_MODEL = PROJECT_ROOT / "Qwen3-0.6B"
SOURCE_DATA = {
    "train": SFT_ROOT / "data" / "code_sft_train.json",
    "valid": SFT_ROOT / "data" / "code_sft_valid.json",
    "test": SFT_ROOT / "data" / "code_sft_test.json",
    "mbpp": SFT_ROOT / "data" / "mbpp_sanitized_test.json",
}
VARIANTS = ("full", "qlora", "qdora")
SCHEMA_VERSION = "a2_v2.1"
SEED = 42


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


def validate_rows(path: Path, required: set[str]) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"{path}[{index}] does not satisfy fields {sorted(required)}")
        for field in (required & {"instruction", "output"}):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"{path}[{index}].{field} is empty")
    return rows


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }


def make_acceptance_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    longest = sorted(
        range(len(rows)),
        key=lambda i: len(str(rows[i]["instruction"])) + len(str(rows[i]["output"])),
        reverse=True,
    )[:32]
    remaining = [i for i in range(len(rows)) if i not in set(longest)]
    random.Random(SEED).shuffle(remaining)
    indices = longest + remaining[:224]
    random.Random(SEED).shuffle(indices)
    return [rows[i] for i in indices]


def deterministic_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:count]]


def symlink_relative(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(os.path.relpath(source, destination.parent))


def base_train_config(profile: str, variant: str, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    is_full = variant == "full"
    config: dict[str, Any] = {
        "model_name_or_path": str(BASE_MODEL),
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full" if is_full else "lora",
        "dataset_dir": str(data_dir),
        "dataset": "a2_train",
        "eval_dataset": "a2_valid",
        "template": "qwen3",
        "enable_thinking": False,
        "cutoff_len": 2048,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 2,
        "output_dir": str(output_dir),
        "logging_steps": 1 if profile == "acceptance" else 10,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "none",
        "skip_memory_metrics": False,
        # The formal profile was calibrated on the 32 longest A1 samples while
        # a 2.7 GiB DPO evaluation shared the GPU. Batch 16 entered vGPU paging
        # and was slower; batch 12 gave the best stable throughput. Keep the
        # smaller settings for the deliberately lightweight acceptance run.
        "per_device_train_batch_size": (2 if is_full else 4) if profile == "acceptance" else 12,
        "gradient_accumulation_steps": (4 if is_full else 2) if profile == "acceptance" else 1,
        "learning_rate": 1.0e-5 if is_full else 1.0e-4,
        "num_train_epochs": 3.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "gradient_checkpointing": True,
        "bf16": True,
        "fp16": False,
        "seed": SEED,
        "data_seed": SEED,
        "ddp_timeout": 180000000,
        "per_device_eval_batch_size": 2,
        "eval_strategy": "steps" if profile == "acceptance" else "epoch",
        "eval_steps": 10 if profile == "acceptance" else None,
        "save_strategy": "no" if profile == "acceptance" else "epoch",
        "save_total_limit": 3,
    }
    if profile == "acceptance":
        config["max_steps"] = 30
    if not is_full:
        config.update(
            {
                "quantization_method": "bnb",
                "quantization_bit": 4,
                "quantization_type": "nf4",
                "double_quantization": True,
                "upcast_layernorm": True,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target": "all",
                "use_dora": variant == "qdora",
            }
        )
    return {key: value for key, value in config.items() if value is not None}


def prepare_run(run_id: str, profile: str, resume: bool) -> dict[str, Any]:
    data_dir = DATA_RUN_ROOT / run_id
    output_dir = OUTPUT_RUN_ROOT / run_id
    manifest_path = data_dir / "run_manifest.json"
    if data_dir.exists() or output_dir.exists():
        if not resume:
            raise FileExistsError(f"RUN_ID already exists: {run_id}; use RESUME=1 to continue")
        if not manifest_path.exists():
            raise RuntimeError(f"Cannot resume without {manifest_path}")
        manifest = read_json(manifest_path)
        if manifest.get("profile") != profile:
            raise RuntimeError(f"Existing profile is {manifest.get('profile')}, requested {profile}")
        return manifest

    train = validate_rows(SOURCE_DATA["train"], {"instruction", "input", "output"})
    valid = validate_rows(SOURCE_DATA["valid"], {"instruction", "input", "output"})
    test = validate_rows(SOURCE_DATA["test"], {"instruction", "input", "output"})
    mbpp = validate_rows(SOURCE_DATA["mbpp"], {"instruction", "input", "output", "task_id"})
    data_dir.mkdir(parents=True)
    for subdir in ("configs", "logs", "models", "merged", "predictions", "eval", "telemetry", "reports", "stages"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    if profile == "acceptance":
        datasets = {
            "a2_train.json": make_acceptance_subset(train),
            "a2_valid.json": deterministic_sample(valid, 64, SEED + 1),
            "a2_test.json": deterministic_sample(test, 32, SEED + 2),
            "a2_mbpp.json": mbpp[:32],
            "a2_mbpp_select.json": mbpp[:32],
        }
        for name, rows in datasets.items():
            write_json(data_dir / name, rows)
    else:
        mapping = {
            "a2_train.json": SOURCE_DATA["train"],
            "a2_valid.json": SOURCE_DATA["valid"],
            "a2_test.json": SOURCE_DATA["test"],
            "a2_mbpp.json": SOURCE_DATA["mbpp"],
        }
        for name, source in mapping.items():
            symlink_relative(source, data_dir / name)
        write_json(data_dir / "a2_mbpp_select.json", mbpp[:64])

    registry = {name.removesuffix(".json"): dataset_entry(name) for name in (
        "a2_train.json", "a2_valid.json", "a2_test.json", "a2_mbpp.json", "a2_mbpp_select.json"
    )}
    write_json(data_dir / "dataset_info.json", registry)

    for variant in VARIANTS:
        config = base_train_config(profile, variant, data_dir, output_dir / "models" / variant)
        with (output_dir / "configs" / f"train_{variant}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    source_manifest = {}
    for name, path in SOURCE_DATA.items():
        rows = read_json(path)
        source_manifest[name] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_id": run_id,
        "profile": profile,
        "project_root": str(PROJECT_ROOT),
        "base_model": str(BASE_MODEL),
        "template": "qwen3",
        "enable_thinking": False,
        "cutoff_len": 2048,
        "seed": SEED,
        "source_data": source_manifest,
        "run_data_dir": str(data_dir),
        "run_output_dir": str(output_dir),
        "variants": list(VARIANTS),
        "acceptance_contract": {"max_steps": 30, "train": 256, "valid": 64, "test": 32, "mbpp": 32},
        "formal_contract": {"num_train_epochs": 3.0, "mbpp_select": 64, "mbpp_final": len(mbpp)},
    }
    write_json(manifest_path, manifest)
    return manifest


def model_spec(output_dir: Path, variant: str, selected: bool = True) -> dict[str, Any]:
    selected_path = output_dir / "selected_models.json"
    if selected and selected_path.exists():
        item = read_json(selected_path)["variants"][variant]
        return item
    path = output_dir / "models" / variant
    if variant == "full":
        return {"finetuning_type": "full", "model_path": str(path), "adapter_path": None}
    return {"finetuning_type": "lora", "model_path": str(BASE_MODEL), "adapter_path": str(path)}


def render_predict(
    run_id: str,
    variant: str,
    dataset: str,
    source_path: str | None,
    tag: str,
    full_model: bool = False,
) -> Path:
    output_dir = OUTPUT_RUN_ROOT / run_id
    data_dir = DATA_RUN_ROOT / run_id
    profile = read_json(data_dir / "run_manifest.json")["profile"]
    spec = (
        {"finetuning_type": "full", "model_path": str(BASE_MODEL), "adapter_path": None}
        if variant == "base"
        else model_spec(output_dir, variant, selected=True)
    )
    if source_path:
        if variant in {"base", "full"} or full_model:
            spec = {"finetuning_type": "full", "model_path": source_path, "adapter_path": None}
        else:
            spec = {"finetuning_type": "lora", "model_path": str(BASE_MODEL), "adapter_path": source_path}
    pred_dir = output_dir / "predictions" / tag / variant
    config: dict[str, Any] = {
        "model_name_or_path": spec["model_path"],
        "trust_remote_code": True,
        "stage": "sft",
        "do_predict": True,
        "finetuning_type": spec["finetuning_type"],
        "dataset_dir": str(data_dir),
        "eval_dataset": dataset,
        "template": "qwen3",
        "enable_thinking": False,
        "cutoff_len": 2048,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "dataloader_num_workers": 2,
        "output_dir": str(pred_dir),
        "overwrite_output_dir": True,
        "report_to": "none",
        "per_device_eval_batch_size": 2,
        "predict_with_generate": True,
        "max_new_tokens": 256 if profile == "acceptance" else 768,
        "do_sample": False,
        "num_beams": 1,
        "bf16": True,
        "fp16": False,
    }
    if spec.get("adapter_path"):
        config["adapter_name_or_path"] = spec["adapter_path"]
    path = output_dir / "configs" / f"predict_{tag}_{variant}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def render_export(run_id: str, variant: str, adapter_path: str | None = None) -> Path:
    if variant not in {"qlora", "qdora"}:
        raise ValueError("Only QLoRA/QDoRA adapters require export")
    output_dir = OUTPUT_RUN_ROOT / run_id
    adapter_path = adapter_path or str(output_dir / "models" / variant)
    config = {
        "model_name_or_path": str(BASE_MODEL),
        "adapter_name_or_path": adapter_path,
        "trust_remote_code": True,
        "finetuning_type": "lora",
        "template": "qwen3",
        "enable_thinking": False,
        "export_dir": str(output_dir / "merged" / variant),
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False,
    }
    path = output_dir / "configs" / f"export_{variant}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def checkpoint_eval_loss(path: Path) -> float:
    state = path / "trainer_state.json"
    if not state.exists():
        return math.inf
    logs = read_json(state).get("log_history", [])
    values = [float(item["eval_loss"]) for item in logs if item.get("eval_loss") is not None]
    return values[-1] if values else math.inf


def select_checkpoints(run_id: str) -> dict[str, Any]:
    output_dir = OUTPUT_RUN_ROOT / run_id
    manifest = read_json(DATA_RUN_ROOT / run_id / "run_manifest.json")
    selected: dict[str, Any] = {"created_at": utc_now(), "run_id": run_id, "variants": {}}
    for variant in VARIANTS:
        model_dir = output_dir / "models" / variant
        if manifest["profile"] == "acceptance":
            candidates = [model_dir]
        else:
            candidates = sorted(model_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if not candidates:
                candidates = [model_dir]
        scored = []
        for candidate in candidates:
            metric_path = output_dir / "eval" / "selection" / variant / candidate.name / "mbpp_metrics.json"
            metrics = read_json(metric_path) if metric_path.exists() else {}
            score = (
                float(metrics.get("pass_at_1", -1.0)),
                float(metrics.get("syntax_pass_rate", -1.0)),
                float(metrics.get("avg_test_pass_rate", -1.0)),
                -checkpoint_eval_loss(candidate),
                int(candidate.name.split("-")[-1]) if candidate.name.startswith("checkpoint-") else 10**12,
            )
            scored.append((score, candidate, metrics))
        _, winner, metrics = max(scored, key=lambda item: item[0])
        selected["variants"][variant] = {
            "training_method": variant,
            "finetuning_type": "full" if variant == "full" else "lora",
            "model_path": str(winner if variant == "full" else BASE_MODEL),
            "adapter_path": None if variant == "full" else str(winner),
            "selected_checkpoint": str(winner),
            "selection_metrics": metrics,
            "merged_model_path": str(winner) if variant == "full" else str(output_dir / "merged" / variant),
        }
    write_json(output_dir / "selected_models.json", selected)
    return selected


FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")


def extract_code(text: str) -> str:
    text = str(text or "").strip()
    matches = FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "class ", "import ", "from ", "@")):
            return "\n".join(lines[i:]).strip()
    return text


def token_f1(left: str, right: str) -> float:
    a, b = TOKEN_RE.findall(left), TOKEN_RE.findall(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def first_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def evaluate_a1(predictions: Path, references: Path, output_dir: Path) -> dict[str, Any]:
    refs = read_json(references)
    items = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = min(len(refs), len(items))
    aggregate = Counter()
    f1_sum = 0.0
    lengths: list[int] = []
    cases = []
    for index in range(total):
        item, ref = items[index], refs[index]
        raw = next((str(item[k]) for k in ("predict", "prediction", "response", "output") if item.get(k)), "")
        code = extract_code(raw)
        reference = str(ref.get("output", "")).strip()
        try:
            ast.parse(code)
            syntax = True
        except SyntaxError:
            syntax = False
        extraction = bool(code and ("def " in code or "class " in code or syntax))
        signature = first_function_name(code) == first_function_name(reference) and first_function_name(reference) is not None
        verbose = bool(re.search(r"(^|\n)(Explanation|Reasoning|Here is|Sure[,!])", raw, re.IGNORECASE))
        aggregate.update(syntax=syntax, extraction=extraction, signature=signature, exact=code == reference, verbose=verbose)
        f1_sum += token_f1(code, reference)
        lengths.append(len(TOKEN_RE.findall(raw)))
        cases.append({"index": index, "instruction": ref.get("instruction"), "predict": raw, "code": code,
                      "syntax_ok": syntax, "signature_ok": signature, "exact": code == reference})
    metrics = {
        "num_tasks": total,
        "syntax_pass_rate": aggregate["syntax"] / total if total else 0.0,
        "code_extraction_rate": aggregate["extraction"] / total if total else 0.0,
        "function_signature_accuracy": aggregate["signature"] / total if total else 0.0,
        "exact_match": aggregate["exact"] / total if total else 0.0,
        "avg_code_token_f1": f1_sum / total if total else 0.0,
        "verbose_output_rate": aggregate["verbose"] / total if total else 0.0,
        "average_generation_tokens": statistics.mean(lengths) if lengths else 0.0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "a1_metrics.json", metrics)
    with (output_dir / "a1_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return metrics


def parse_telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8"), skipinitialspace=True))
    values = []
    for row in rows:
        raw = row.get("memory.used [MiB]", row.get("memory.used", ""))
        match = re.search(r"\d+(?:\.\d+)?", str(raw))
        if match:
            values.append(float(match.group()))
    return {"peak_gpu_memory_mib": max(values) if values else None, "samples": len(values)}


def trainer_summary(model_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    train_results = model_dir / "train_results.json"
    if train_results.exists():
        result.update(read_json(train_results))
    state = model_dir / "trainer_state.json"
    if state.exists():
        history = read_json(state).get("log_history", [])
        losses = [float(item["loss"]) for item in history if item.get("loss") is not None and math.isfinite(float(item["loss"]))]
        gradients = [float(item["grad_norm"]) for item in history if item.get("grad_norm") is not None]
        result["optimizer_steps_logged"] = len(losses)
        result["first_loss_window"] = statistics.mean(losses[:5]) if losses else None
        result["last_loss_window"] = statistics.mean(losses[-5:]) if losses else None
        result["finite_grad_norm_ratio"] = sum(math.isfinite(x) for x in gradients) / len(gradients) if gradients else None
    return result


def parse_trainable_params(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)\s*\|\|\s*trainable%:\s*([\d.]+)",
        text,
    )
    if not match:
        return {}
    return {
        "trainable_parameters": int(match.group(1).replace(",", "")),
        "all_parameters": int(match.group(2).replace(",", "")),
        "trainable_parameter_percent": float(match.group(3)),
    }


def validate_artifact_changes(run_id: str) -> dict[str, Any]:
    from safetensors import safe_open

    output_dir = OUTPUT_RUN_ROOT / run_id
    profile = read_json(DATA_RUN_ROOT / run_id / "run_manifest.json")["profile"]
    results: dict[str, Any] = {"created_at": utc_now(), "run_id": run_id, "variants": {}}
    base_file = BASE_MODEL / "model.safetensors"
    for variant in VARIANTS:
        model_dir = output_dir / "models" / variant
        state_path = model_dir / "trainer_state.json"
        state = read_json(state_path) if state_path.exists() else {}
        global_step = int(state.get("global_step", 0))
        if profile == "acceptance" and global_step != 30:
            raise RuntimeError(f"{variant} finished at global_step={global_step}, expected 30")
        item: dict[str, Any] = {"global_step": global_step, "changed": False}
        if variant == "full":
            trained_file = model_dir / "model.safetensors"
            if not base_file.exists() or not trained_file.exists():
                raise FileNotFoundError(f"Missing full-model safetensors for comparison: {trained_file}")
            with safe_open(base_file, framework="pt", device="cpu") as base, safe_open(
                trained_file, framework="pt", device="cpu"
            ) as trained:
                common = [key for key in trained.keys() if key in set(base.keys()) and "embed_tokens" not in key][:8]
                deltas = [float((trained.get_tensor(key).float() - base.get_tensor(key).float()).abs().max()) for key in common]
            item.update({"sampled_tensors": len(deltas), "max_parameter_delta": max(deltas, default=0.0)})
            item["changed"] = item["max_parameter_delta"] > 0
        else:
            adapter_file = model_dir / "adapter_model.safetensors"
            if not adapter_file.exists():
                raise FileNotFoundError(f"Missing adapter weights: {adapter_file}")
            with safe_open(adapter_file, framework="pt", device="cpu") as adapter:
                b_keys = [key for key in adapter.keys() if "lora_B" in key]
                nonzero = sum(int(bool(adapter.get_tensor(key).abs().max().item() > 0)) for key in b_keys)
            item.update({"lora_b_tensors": len(b_keys), "nonzero_lora_b_tensors": nonzero})
            item["changed"] = bool(b_keys) and nonzero > 0
        if not item["changed"]:
            raise RuntimeError(f"No trainable parameter change detected for {variant}")
        results["variants"][variant] = item
    write_json(output_dir / "reports" / "artifact_change_validation.json", results)
    return results


def build_report(run_id: str) -> dict[str, Any]:
    output_dir = OUTPUT_RUN_ROOT / run_id
    manifest = read_json(DATA_RUN_ROOT / run_id / "run_manifest.json")
    selected = read_json(output_dir / "selected_models.json") if (output_dir / "selected_models.json").exists() else select_checkpoints(run_id)
    variants: dict[str, Any] = {}
    for variant in ("base",) + VARIANTS:
        entry: dict[str, Any] = {}
        if variant != "base":
            entry["training"] = trainer_summary(output_dir / "models" / variant)
            entry["training"].update(parse_trainable_params(output_dir / "logs" / f"train_{variant}.log"))
            torch_telemetry = output_dir / "telemetry" / f"train_{variant}_torch.json"
            entry["telemetry"] = read_json(torch_telemetry) if torch_telemetry.exists() else parse_telemetry(
                output_dir / "telemetry" / f"train_{variant}.csv"
            )
        mbpp = output_dir / "eval" / "final" / variant / "mbpp_metrics.json"
        a1 = output_dir / "eval" / "final" / variant / "a1_metrics.json"
        entry["mbpp"] = read_json(mbpp) if mbpp.exists() else {}
        entry["a1"] = read_json(a1) if a1.exists() else {}
        predict_results = output_dir / "predictions" / "final_mbpp" / variant / "predict_results.json"
        if predict_results.exists():
            prediction = read_json(predict_results)
            runtime = prediction.get("predict_runtime")
            tasks = entry["mbpp"].get("num_tasks")
            entry["inference"] = {
                "runtime_seconds": runtime,
                "average_latency_seconds": runtime / tasks if runtime and tasks else None,
                "samples_per_second": prediction.get("predict_samples_per_second"),
            }
        variants[variant] = entry
    report = {"schema_version": SCHEMA_VERSION, "created_at": utc_now(), "run_id": run_id,
              "profile": manifest["profile"], "variants": variants, "selected_models": selected["variants"]}
    bridge_summary = output_dir / "a4_bridge" / "bridge_summary.json"
    if bridge_summary.exists():
        report["a2_a3_a4_bridge"] = read_json(bridge_summary)
    report_dir = output_dir / "reports"
    write_json(report_dir / "comparison_report.json", report)
    columns = ["variant", "pass_at_1", "syntax_pass_rate", "avg_test_pass_rate", "a1_token_f1", "peak_gpu_memory_mib", "train_runtime"]
    with (report_dir / "comparison_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for variant, item in variants.items():
            writer.writerow({
                "variant": variant,
                "pass_at_1": item["mbpp"].get("pass_at_1"),
                "syntax_pass_rate": item["mbpp"].get("syntax_pass_rate"),
                "avg_test_pass_rate": item["mbpp"].get("avg_test_pass_rate"),
                "a1_token_f1": item["a1"].get("avg_code_token_f1"),
                "peak_gpu_memory_mib": item.get("telemetry", {}).get("peak_gpu_memory_mib"),
                "train_runtime": item.get("training", {}).get("train_runtime"),
            })
    lines = [f"# A2 v2 comparison: {run_id}", "", f"Profile: `{manifest['profile']}`", "",
             "| Variant | MBPP pass@1 | Syntax | Avg test pass | A1 token F1 | Peak GPU MiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, item in variants.items():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            variant, item["mbpp"].get("pass_at_1", ""), item["mbpp"].get("syntax_pass_rate", ""),
            item["mbpp"].get("avg_test_pass_rate", ""), item["a1"].get("avg_code_token_f1", ""),
            item.get("telemetry", {}).get("peak_gpu_memory_mib", "")))
    (report_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        labels = list(variants)
        pass_at_1 = [variants[x]["mbpp"].get("pass_at_1", 0.0) for x in labels]
        memory = [variants[x].get("telemetry", {}).get("peak_gpu_memory_mib") or 0.0 for x in labels]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, pass_at_1)
        axes[0].set_title("MBPP pass@1")
        axes[0].set_ylim(0, max(1.0, max(pass_at_1, default=0.0) * 1.2))
        axes[1].bar(labels, memory)
        axes[1].set_title("Peak training GPU memory (MiB)")
        fig.tight_layout()
        fig.savefig(report_dir / "comparison_metrics.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        write_json(report_dir / "plot_warning.json", {"warning": str(exc)})

    handoff = {
        "schema_version": "a2_handoff_v1",
        "created_at": utc_now(),
        "run_id": run_id,
        "profile": manifest["profile"],
        "base_model": str(BASE_MODEL),
        "training_contract": {"template": "qwen3", "enable_thinking": False, "cutoff_len": 2048},
        "a1_source_data": manifest["source_data"],
        "variants": selected["variants"],
        "comparison_report": str(report_dir / "comparison_report.json"),
    }
    write_json(output_dir / "a2_handoff_manifest.json", handoff)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--profile", choices=("acceptance", "formal"), required=True)
    prepare.add_argument("--resume", action="store_true")
    render_predict_parser = sub.add_parser("render-predict")
    render_predict_parser.add_argument("--run-id", required=True)
    render_predict_parser.add_argument("--variant", choices=("base",) + VARIANTS, required=True)
    render_predict_parser.add_argument("--dataset", required=True)
    render_predict_parser.add_argument("--source-path")
    render_predict_parser.add_argument("--tag", required=True)
    render_predict_parser.add_argument("--full-model", action="store_true")
    render_export_parser = sub.add_parser("render-export")
    render_export_parser.add_argument("--run-id", required=True)
    render_export_parser.add_argument("--variant", choices=("qlora", "qdora"), required=True)
    render_export_parser.add_argument("--adapter-path")
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--run-id", required=True)
    a1_parser = sub.add_parser("evaluate-a1")
    a1_parser.add_argument("--predictions", type=Path, required=True)
    a1_parser.add_argument("--references", type=Path, required=True)
    a1_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--run-id", required=True)
    validate_parser = sub.add_parser("validate-artifacts")
    validate_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        print(json.dumps(prepare_run(args.run_id, args.profile, args.resume), indent=2))
    elif args.command == "render-predict":
        print(render_predict(args.run_id, args.variant, args.dataset, args.source_path, args.tag, args.full_model))
    elif args.command == "render-export":
        print(render_export(args.run_id, args.variant, args.adapter_path))
    elif args.command == "select":
        print(json.dumps(select_checkpoints(args.run_id), indent=2))
    elif args.command == "evaluate-a1":
        print(json.dumps(evaluate_a1(args.predictions, args.references, args.output_dir), indent=2))
    elif args.command == "report":
        print(json.dumps(build_report(args.run_id), indent=2))
    elif args.command == "validate-artifacts":
        print(json.dumps(validate_artifact_changes(args.run_id), indent=2))


if __name__ == "__main__":
    main()
