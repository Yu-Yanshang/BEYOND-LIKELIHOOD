"""Discover and normalize evaluation/training metrics across /root/project."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path("/root/project")


@dataclass
class MetricRecord:
    kind: str
    module: str
    label: str
    relative_path: str
    training_method: str
    strategy: str
    protocol: str
    benchmark: str
    tasks: int | None = None
    passed_tasks: int | None = None
    pass_at_1: float | None = None
    syntax_pass_rate: float | None = None
    avg_test_pass_rate: float | None = None
    exact_match: float | None = None
    code_token_f1: float | None = None
    latency_seconds: float | None = None
    candidates: int | None = None
    epoch: float | None = None
    train_loss: float | None = None
    eval_loss: float | None = None
    train_runtime: float | None = None
    is_smoke: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def first_number(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def infer_module(relative: Path) -> str:
    return relative.parts[0] if relative.parts else "project"


def infer_training_method(path_text: str, data: dict[str, Any]) -> str:
    path_lower = path_text.lower()
    leaf_directory = path_lower.rsplit("/", 1)[0].rsplit("/", 1)[-1]
    if leaf_directory in ("full_dpo", "full_dpo_epoch_continue"):
        return "Full DPO"
    if leaf_directory in ("lora_dpo", "lora_dpo_epoch_continue", "a4_dpo_adapter"):
        return "LoRA DPO"
    if leaf_directory in ("ppo_adapter", "ppo_reward", "ppo_reward_fn_epoch_continue"):
        return "PPO/GRPO"
    if "best_path/" in path_lower:
        if "/a5_from_a2_full/" in path_lower:
            return "A2 Full + A5"
        if "/a5_from_a4_dpo/" in path_lower:
            return "A4 DPO + A5"
        if "/a2_full/" in path_lower or "/models/a2_full/" in path_lower:
            return "Full SFT"
        if "/a4_dpo" in path_lower or "/models/a4_dpo" in path_lower:
            return "LoRA DPO"
    if "/mbpp_eval/" in path_lower:
        variant = path_lower.split("/mbpp_eval/", 1)[1].split("/", 1)[0]
        if "ppo" in variant or "grpo" in variant:
            return "PPO/GRPO"
        if "lora_dpo" in variant:
            return "LoRA DPO"
        if "full_dpo" in variant or variant in ("dpo", "dpo_final"):
            return "Full DPO"
        if variant.startswith("a2_pref"):
            return "DPO"
        if variant in ("a2_full", "base"):
            return "Full SFT"
    if "mbpp_eval_qwen3_base" in path_lower or "mbpp_base_dpo_compare/base" in path_lower:
        return "Base"
    if "mbpp_eval_qwen3_06b_lora_dpo" in path_lower:
        return "LoRA DPO"
    if "mbpp_eval_qwen3_dpo" in path_lower:
        return "Full DPO"
    if path_lower.startswith("tts/") and str(data.get("model_path", "")).lower().rstrip("/").endswith("/a2_full"):
        return "Full SFT"
    text = " ".join(
        str(value).lower() for value in (path_lower, data.get("model_path", ""), data.get("adapter_path", ""))
    )
    if "ppo" in text or "grpo" in text:
        return "PPO/GRPO"
    if "qdora" in text:
        return "QDoRA SFT"
    if "qlora" in text:
        return "QLoRA SFT"
    if "lora_dpo" in text or "dpo_adapter" in text:
        return "LoRA DPO"
    if "full_dpo" in text or "code_full_dpo" in text:
        return "Full DPO"
    if "a4_dpo" in text or "/dpo" in text or "dpo/" in text:
        return "DPO"
    if "a5_from_a2" in text:
        return "A2 Full + A5"
    if "a5_from_a4" in text:
        return "A4 DPO + A5"
    if "a2_full" in text or "/full/" in text or "models/full" in text:
        return "Full SFT"
    if "/base/" in text or text.endswith("base") or "qwen3_base" in text:
        return "Base"
    if "sft" in text:
        return "SFT"
    return "Unspecified"


def infer_strategy(path_text: str, data: dict[str, Any]) -> str:
    if data.get("strategy"):
        return str(data["strategy"])
    text = path_text.lower()
    for needle, label in (
        ("self_consistency", "self_consistency"),
        ("best_of_n", "best_of_n"),
        ("reflexion", "reflexion"),
        ("/tot/", "tot"),
        ("cot_code", "cot"),
    ):
        if needle in text:
            return label
    prompt = data.get("prompt")
    if isinstance(prompt, dict) and prompt.get("mode"):
        return str(prompt["mode"])
    if data.get("predictions"):
        return "single_generation"
    return "training" if "train_results" in text or "eval_results" in text else "unspecified"


def infer_protocol(relative: Path, data: dict[str, Any]) -> str:
    text = relative.as_posix().lower()
    if "best_path" in text:
        if "legacy_protocol" in text:
            return "Google MBPP legacy"
        return "best_path visible-tests"
    if "tts/outputs/cot_code_eval" in text:
        return "legacy CoT sections"
    if "tts/outputs/tts_" in text:
        return "TTS v2 visible-tests"
    prompt = data.get("prompt")
    if isinstance(prompt, dict) and "google research" in str(prompt.get("source", "")).lower():
        return "Google MBPP legacy"
    if relative.parts and relative.parts[0] == "sft":
        if "a1_metrics" in text:
            return "A1 held-out generation"
        return "LLaMA-Factory prediction"
    if relative.parts and relative.parts[0] == "dpo":
        return "DPO training" if "train_results" in text or "eval_results" in text else "DPO evaluation"
    return "training log" if "results.json" in text else "unspecified"


def infer_label(relative: Path) -> str:
    parts = list(relative.parts)
    text = relative.as_posix()
    if "best_path" in parts and "eval" in parts:
        index = parts.index("eval")
        return "best/" + "/".join(parts[index + 1 : -1])
    if parts and parts[0] == "tts" and "outputs" in parts:
        index = parts.index("outputs")
        return "tts/" + "/".join(parts[index + 1 : -1])
    if parts and parts[0] == "sft" and "eval" in parts:
        index = parts.index("eval")
        return "sft/" + "/".join(parts[index + 1 : -1])
    if parts and parts[0] == "dpo" and "mbpp_eval" in parts:
        index = parts.index("mbpp_eval")
        run = parts[2] if len(parts) > 2 else "run"
        return f"dpo/{run}/" + "/".join(parts[index + 1 : -1])
    if relative.name in ("train_results.json", "eval_results.json"):
        return "/".join(parts[-4:-1])
    return text.removesuffix("/metrics.json").removesuffix("/mbpp_metrics.json")


def infer_benchmark(data: dict[str, Any], relative: Path) -> str:
    benchmark = str(data.get("benchmark", ""))
    config = str(data.get("config", ""))
    if benchmark and config and config.lower() not in benchmark.lower():
        return f"{benchmark} {config}"
    if benchmark:
        return benchmark
    input_file = str(data.get("input_file", "")).lower()
    if "mbpp" in input_file and "sanitized" in input_file:
        return "MBPP sanitized"
    if "a1_metrics" in relative.name:
        return "A1 held-out"
    if "train_results" in relative.name:
        return "training"
    if "eval_results" in relative.name:
        return "validation loss"
    return "unspecified"


def parse_record(project_root: Path, path: Path) -> MetricRecord | None:
    data = load_json(path)
    if not data:
        return None
    relative = path.relative_to(project_root)
    text = relative.as_posix()
    tasks_value = first_number(data, "total", "num_tasks")
    tasks = int(tasks_value) if tasks_value is not None else None
    passed_value = first_number(data, "passed_tasks")
    passed_tasks = int(passed_value) if passed_value is not None else None
    pass_at_1 = first_number(data, "pass_at_1")
    syntax = first_number(data, "syntax_pass_rate")
    train_loss = first_number(data, "train_loss")
    eval_loss = first_number(data, "eval_loss")
    if pass_at_1 is not None or syntax is not None or "a1_metrics" in path.name:
        kind = "evaluation"
    elif train_loss is not None:
        kind = "training"
    elif eval_loss is not None:
        kind = "validation"
    else:
        return None
    is_smoke = bool(
        (tasks is not None and tasks < 200)
        or any(token in text.lower() for token in ("smoke", "probe", "accept"))
    )
    return MetricRecord(
        kind=kind,
        module=infer_module(relative),
        label=infer_label(relative),
        relative_path=text,
        training_method=infer_training_method(text, data),
        strategy=infer_strategy(text, data),
        protocol=infer_protocol(relative, data),
        benchmark=infer_benchmark(data, relative),
        tasks=tasks,
        passed_tasks=passed_tasks,
        pass_at_1=pass_at_1,
        syntax_pass_rate=syntax,
        avg_test_pass_rate=first_number(data, "avg_test_pass_rate"),
        exact_match=first_number(data, "exact_match"),
        code_token_f1=first_number(data, "avg_code_token_f1"),
        latency_seconds=first_number(data, "average_latency_seconds", "seconds_per_task"),
        candidates=int(first_number(data, "candidates", "num_samples", "best_of_n") or 0) or None,
        epoch=first_number(data, "epoch"),
        train_loss=train_loss,
        eval_loss=eval_loss,
        train_runtime=first_number(data, "train_runtime"),
        is_smoke=is_smoke,
    )


def discover_metric_records(project_root: Path = PROJECT_ROOT) -> list[MetricRecord]:
    paths = set(project_root.rglob("*metrics.json"))
    paths.update(project_root.rglob("train_results.json"))
    paths.update(project_root.rglob("eval_results.json"))
    records = []
    for path in sorted(paths):
        if any(part in ("LlamaFactory", ".git", "__pycache__") for part in path.parts):
            continue
        record = parse_record(project_root, path)
        if record is not None:
            records.append(record)
    return records


def full_evaluations(records: Iterable[MetricRecord], include_smoke: bool = False) -> list[MetricRecord]:
    values = [record for record in records if record.kind == "evaluation"]
    if not include_smoke:
        values = [record for record in values if not record.is_smoke]
    return values


def merged_training_runs(records: Iterable[MetricRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.kind not in ("training", "validation"):
            continue
        directory = str(Path(record.relative_path).parent)
        item = grouped.setdefault(
            directory,
            {
                "module": record.module,
                "label": directory,
                "training_method": record.training_method,
                "epoch": None,
                "train_loss": None,
                "eval_loss": None,
                "train_runtime": None,
            },
        )
        for key in ("epoch", "train_loss", "eval_loss", "train_runtime"):
            value = getattr(record, key)
            if value is not None:
                item[key] = value
    return sorted(grouped.values(), key=lambda item: (item["module"], item["label"]))


def running_evaluation_jobs() -> list[str]:
    jobs = []
    proc = Path("/proc")
    if not proc.exists():
        return jobs
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if int(entry.name) == os.getpid():
            continue
        if "test_time_scaling.py" in command:
            jobs.append(f"PID {entry.name}: {command.strip()}")
    return jobs


def export_catalog(records: Iterable[MetricRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]
