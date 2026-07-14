#!/usr/bin/env python3
"""A3 v2 preference-data construction for Qwen3 Python-code DPO.

This module is intentionally additive: it writes versioned A3 v2 run directories and
never rewrites the historical dpo/data/code_dpo_train.json, code_dpo_test.json, or
dataset_info.json files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "py-dpo-v0.1"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dpo" / "data" / "a3_v2_runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dpo" / "outputs" / "a3_v2_runs"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Qwen3-0.6B"
DEFAULT_PROMPT_TEMPLATE = (
    "Complete the following Python coding task. Return a correct, readable "
    "Python solution and include brief reasoning when helpful.\n\nTask:\n{prompt}"
)

TRAIN_COLUMNS = {"instruction", "input", "chosen", "rejected"}
TEST_COLUMNS = {"instruction", "input", "output"}
SCORE_VERSION = "a3_v2_quality_score_1.0"
DEFAULT_QUALITY_THRESHOLD = 0.72
DEFAULT_DEBUG_SIZE = 128

SCORE_WEIGHTS = {
    "integrity_score": 0.20,
    "chosen_code_score": 0.25,
    "separability_score": 0.25,
    "length_safety_score": 0.15,
    "length_balance_score": 0.10,
    "format_score": 0.05,
}

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_CODE_START_RE = re.compile(
    r"^(def\s+|async\s+def\s+|class\s+|from\s+\S+\s+import\s+|import\s+|@|if\s+__name__\s*==|for\s+|while\s+|try\s*:|with\s+)",
    re.IGNORECASE,
)
_CODE_LIKE_RE = re.compile(r"\b(def|class|return|import|from|for|while|if|elif|else|try|except|lambda)\b|[A-Za-z_]\w*\s*=")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id() -> str:
    return "a3_v2_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).strip()


def stable_hash(*parts: Any, length: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(clean_text(part).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:length]


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def build_instruction(prompt: str, prompt_template: str) -> str:
    return prompt_template.format(prompt=prompt).strip()


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception:
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        except Exception:
            try:
                from datasets import load_dataset

                dataset = load_dataset("parquet", data_files=str(path), split="train")
                return [dict(row) for row in dataset]
            except Exception as datasets_error:
                raise RuntimeError(
                    "Failed to read parquet. Install pandas, pyarrow, or datasets with parquet support."
                ) from datasets_error


def find_parquet_files(source_dir: Path) -> list[Path]:
    candidates = sorted(source_dir.glob("*.parquet"))
    data_dir = source_dir / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("*.parquet")))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")
    return candidates


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_raw_records(source_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    raw_index = 0
    for parquet_file in find_parquet_files(source_dir):
        rows = read_parquet(parquet_file)
        source_files.append({"path": str(parquet_file), "sha256": sha256_file(parquet_file), "rows": len(rows)})
        for source_row, row in enumerate(rows):
            item = dict(row)
            item["_source_file"] = rel_to_project(parquet_file)
            item["_source_row"] = source_row
            item["_raw_index"] = raw_index
            records.append(item)
            raw_index += 1
    return records, source_files


@dataclass
class TokenLengthEstimator:
    model_path: Path | None
    tokenizer: Any = None
    mode: str = "char_proxy"

    @classmethod
    def create(cls, model_path: Path | None) -> "TokenLengthEstimator":
        estimator = cls(model_path=model_path)
        if model_path is None:
            return estimator
        try:
            from transformers import AutoTokenizer

            estimator.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True, local_files_only=True
            )
            estimator.mode = "qwen3_tokenizer"
        except Exception as exc:
            estimator.tokenizer = None
            estimator.mode = f"char_proxy_tokenizer_unavailable:{type(exc).__name__}"
        return estimator

    def count(self, text: str) -> int:
        text = clean_text(text)
        if not text:
            return 0
        if self.tokenizer is not None:
            return int(len(self.tokenizer.encode(text, add_special_tokens=False)))
        return max(1, math.ceil(len(text) / 4))


def extract_code(response: str) -> dict[str, Any]:
    text = clean_text(response)
    blocks = [block.strip() for block in _CODE_FENCE_RE.findall(text) if block.strip()]
    if blocks:
        code = max(blocks, key=len)
        return {"code": code, "has_fence": True, "extractable": True, "strategy": "largest_fenced_block"}

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if _CODE_START_RE.search(line.strip()):
            code = "\n".join(lines[idx:]).strip()
            return {"code": code, "has_fence": False, "extractable": bool(code), "strategy": "from_first_code_like_line"}

    return {
        "code": text,
        "has_fence": False,
        "extractable": bool(_CODE_LIKE_RE.search(text)),
        "strategy": "full_text_fallback",
    }


def syntax_ok(code: str) -> tuple[bool, str]:
    if not clean_text(code):
        return False, "empty_code"
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError:{exc.msg}"
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def code_tokens(code: str) -> list[str]:
    try:
        toks: list[str] = []
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type in {
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.ENCODING,
            }:
                continue
            toks.append(tok.string)
        if toks:
            return toks[:5000]
    except Exception:
        pass
    return re.findall(r"\w+|[^\s\w]", code)[:5000]


def pair_similarity(chosen_code: str, rejected_code: str) -> float:
    chosen_tokens = code_tokens(chosen_code)
    rejected_tokens = code_tokens(rejected_code)
    if not chosen_tokens and not rejected_tokens:
        return 1.0
    if not chosen_tokens or not rejected_tokens:
        return 0.0
    return float(SequenceMatcher(None, chosen_tokens, rejected_tokens, autojunk=False).ratio())


def score_pair(instruction: str, chosen: str, rejected: str, estimator: TokenLengthEstimator) -> dict[str, Any]:
    norm_chosen = normalize_ws(chosen)
    norm_rejected = normalize_ws(rejected)

    chosen_extract = extract_code(chosen)
    rejected_extract = extract_code(rejected)
    chosen_syntax, chosen_syntax_error = syntax_ok(chosen_extract["code"])
    rejected_syntax, rejected_syntax_error = syntax_ok(rejected_extract["code"])
    similarity = pair_similarity(chosen_extract["code"], rejected_extract["code"])

    prompt_tokens = estimator.count(instruction)
    chosen_response_tokens = estimator.count(chosen)
    rejected_response_tokens = estimator.count(rejected)
    chosen_total_tokens = prompt_tokens + chosen_response_tokens
    rejected_total_tokens = prompt_tokens + rejected_response_tokens

    chosen_equals_rejected = bool(norm_chosen and norm_chosen == norm_rejected)
    integrity_component = 1.0 if instruction and chosen and rejected and not chosen_equals_rejected else 0.0
    chosen_code_component = (0.10 if chosen_extract["extractable"] else 0.0) + (0.15 if chosen_syntax else 0.0)
    separability_component = min(1.0, max(0.0, (1.0 - similarity) / 0.50))

    too_short = chosen_response_tokens < 8 or rejected_response_tokens < 8
    over_4096 = chosen_total_tokens > 4096 or rejected_total_tokens > 4096
    over_2048_count = int(chosen_total_tokens > 2048) + int(rejected_total_tokens > 2048)
    if too_short or over_4096:
        length_safety_component = 0.0
    elif over_2048_count == 0:
        length_safety_component = 1.0
    elif over_2048_count == 1:
        length_safety_component = 0.5
    else:
        length_safety_component = 0.25

    max_resp = max(chosen_response_tokens, rejected_response_tokens, 1)
    min_resp = max(min(chosen_response_tokens, rejected_response_tokens), 0)
    length_balance_component = min_resp / max_resp
    format_component = 1.0 if chosen_extract["has_fence"] or chosen_extract["extractable"] else 0.0

    components = {
        "integrity_score": round(integrity_component * SCORE_WEIGHTS["integrity_score"], 6),
        "chosen_code_score": round(chosen_code_component, 6),
        "separability_score": round(separability_component * SCORE_WEIGHTS["separability_score"], 6),
        "length_safety_score": round(length_safety_component * SCORE_WEIGHTS["length_safety_score"], 6),
        "length_balance_score": round(length_balance_component * SCORE_WEIGHTS["length_balance_score"], 6),
        "format_score": round(format_component * SCORE_WEIGHTS["format_score"], 6),
    }
    total_score = round(sum(components.values()), 6)

    flags = {
        "chosen_equals_rejected": chosen_equals_rejected,
        "near_duplicate": similarity >= 0.95,
        "low_separation": similarity >= 0.80,
        "chosen_syntax_error": not chosen_syntax,
        "rejected_syntax_error": not rejected_syntax,
        "chosen_too_long": chosen_total_tokens > 2048,
        "rejected_too_long": rejected_total_tokens > 2048,
        "extreme_length_gap": (max_resp / max(min_resp, 1)) >= 4.0,
        "only_rejected_syntax_ok": (not chosen_syntax) and rejected_syntax,
        "both_syntax_error": (not chosen_syntax) and (not rejected_syntax),
        "response_too_short": too_short,
    }

    return {
        "quality_score": total_score,
        "score_version": SCORE_VERSION,
        "score_components": components,
        "flags": flags,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "chosen_response_tokens": chosen_response_tokens,
            "rejected_response_tokens": rejected_response_tokens,
            "chosen_total_tokens": chosen_total_tokens,
            "rejected_total_tokens": rejected_total_tokens,
            "chosen_minus_rejected_response_tokens": chosen_response_tokens - rejected_response_tokens,
            "chosen_code_chars": len(chosen_extract["code"]),
            "rejected_code_chars": len(rejected_extract["code"]),
            "pair_similarity": round(similarity, 6),
            "chosen_has_fence": chosen_extract["has_fence"],
            "rejected_has_fence": rejected_extract["has_fence"],
            "chosen_extractable": chosen_extract["extractable"],
            "rejected_extractable": rejected_extract["extractable"],
            "chosen_syntax_ok": chosen_syntax,
            "rejected_syntax_ok": rejected_syntax,
            "chosen_syntax_error_message": chosen_syntax_error,
            "rejected_syntax_error_message": rejected_syntax_error,
            "chosen_code_strategy": chosen_extract["strategy"],
            "rejected_code_strategy": rejected_extract["strategy"],
        },
    }


def train_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("instruction", ""),
        row.get("input", ""),
        row.get("chosen", ""),
        row.get("rejected", ""),
        row.get("output", ""),
    )


def test_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("instruction", ""), row.get("input", ""), row.get("output", ""))


def source_prompt_and_group(raw: dict[str, Any]) -> tuple[str, str, str]:
    prompt = first_text(raw, ("prompt", "instruction", "question"))
    source_id = first_text(raw, ("id", "sample_id", "uid"))
    group_id = source_id or stable_hash("prompt", normalize_ws(prompt), length=20)
    return prompt, source_id, group_id


def make_train_row(raw: dict[str, Any], prompt_template: str) -> dict[str, str] | None:
    prompt = first_text(raw, ("prompt", "instruction", "question"))
    chosen = first_text(raw, ("chosen", "accepted", "preferred", "output"))
    rejected = first_text(raw, ("rejected", "reject", "dispreferred"))
    input_text = first_text(raw, ("input", "query"))
    if not prompt or not chosen or not rejected:
        return None
    return {
        "instruction": build_instruction(prompt, prompt_template),
        "input": input_text,
        "chosen": chosen,
        "rejected": rejected,
    }


def make_test_row(raw: dict[str, Any], prompt_template: str) -> dict[str, str] | None:
    prompt = first_text(raw, ("prompt", "instruction", "question"))
    output = first_text(raw, ("chosen", "accepted", "preferred", "output"))
    input_text = first_text(raw, ("input", "query"))
    if not prompt or not output:
        return None
    return {
        "instruction": build_instruction(prompt, prompt_template),
        "input": input_text,
        "output": output,
    }


def provenance(raw: dict[str, Any], split: str) -> dict[str, Any]:
    prompt, source_id, group_id = source_prompt_and_group(raw)
    return {
        "split": split,
        "raw_index": raw.get("_raw_index"),
        "source_file": raw.get("_source_file"),
        "source_row": raw.get("_source_row"),
        "source_id": source_id,
        "group_id": group_id,
        "prompt_sha256_16": stable_hash("prompt", normalize_ws(prompt), length=16),
    }


def build_train_views(
    raw_train_rows: list[dict[str, Any]],
    *,
    prompt_template: str,
    estimator: TokenLengthEstimator,
    seed: int,
    quality_threshold: float,
    debug_size: int,
) -> dict[str, Any]:
    seen: set[tuple[str, str, str, str, str]] = set()
    raw_items: list[tuple[dict[str, str], dict[str, Any]]] = []
    clean_items: list[tuple[dict[str, str], dict[str, Any]]] = []
    high_items: list[tuple[dict[str, str], dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []

    for raw in raw_train_rows:
        prompt, source_id, _group_id = source_prompt_and_group(raw)
        sample_id = stable_hash("train", raw.get("_raw_index"), source_id, prompt, raw.get("chosen"), raw.get("rejected"), length=20)
        audit: dict[str, Any] = {
            "sample_id": sample_id,
            **provenance(raw, "train"),
            "views": [],
            "drop_stage": None,
            "drop_reasons": [],
        }
        row = make_train_row(raw, prompt_template)
        if row is None:
            audit["drop_stage"] = "raw_conversion"
            audit["drop_reasons"].append("field_missing")
            audits.append(audit)
            continue

        key = train_key(row)
        if key in seen:
            audit["drop_stage"] = "raw_conversion"
            audit["drop_reasons"].append("duplicate_in_split")
            audits.append(audit)
            continue
        seen.add(key)

        quality = score_pair(row["instruction"], row["chosen"], row["rejected"], estimator)
        audit.update(quality)
        audit["views"].append("raw_train")
        audit["drop_reasons"].extend([name for name, value in quality["flags"].items() if value])
        metadata = {
            **audit,
            "row_hash": stable_hash(row["instruction"], row["input"], row["chosen"], row["rejected"], length=20),
        }
        raw_items.append((row, metadata))

        clean_ok = not quality["flags"]["chosen_equals_rejected"]
        if clean_ok:
            metadata["views"].append("clean_train")
            clean_items.append((row, metadata))
            if quality["quality_score"] >= quality_threshold:
                metadata["views"].append("high_confidence_train")
                high_items.append((row, metadata))
            else:
                metadata["drop_stage"] = "high_confidence"
                metadata["drop_reasons"].append("below_quality_threshold")
        else:
            metadata["drop_stage"] = "clean"

        audits.append(metadata)

    rng = random.Random(seed)
    debug_items = list(high_items)
    if len(debug_items) > debug_size:
        debug_items = rng.sample(debug_items, debug_size)

    return {
        "raw_items": raw_items,
        "clean_items": clean_items,
        "high_items": high_items,
        "debug_items": debug_items,
        "audits": audits,
    }


def build_generation_test(raw_test_rows: list[dict[str, Any]], *, prompt_template: str) -> dict[str, Any]:
    seen: set[tuple[str, str, str]] = set()
    items: list[tuple[dict[str, str], dict[str, Any]]] = []
    audits: list[dict[str, Any]] = []
    for raw in raw_test_rows:
        prompt, source_id, _group_id = source_prompt_and_group(raw)
        sample_id = stable_hash("test", raw.get("_raw_index"), source_id, prompt, raw.get("chosen"), length=20)
        audit: dict[str, Any] = {
            "sample_id": sample_id,
            **provenance(raw, "test"),
            "views": [],
            "drop_stage": None,
            "drop_reasons": [],
        }
        row = make_test_row(raw, prompt_template)
        if row is None:
            audit["drop_stage"] = "test_conversion"
            audit["drop_reasons"].append("field_missing")
            audits.append(audit)
            continue
        key = test_key(row)
        if key in seen:
            audit["drop_stage"] = "test_conversion"
            audit["drop_reasons"].append("duplicate_in_split")
            audits.append(audit)
            continue
        seen.add(key)
        audit["views"].append("generation_test")
        metadata = {
            **audit,
            "row_hash": stable_hash(row["instruction"], row["input"], row["output"], length=20),
        }
        items.append((row, metadata))
        audits.append(metadata)
    return {"items": items, "audits": audits}


def rows_only(items: list[tuple[dict[str, str], dict[str, Any]]]) -> list[dict[str, str]]:
    return [row for row, _meta in items]


def metas_only(items: list[tuple[dict[str, str], dict[str, Any]]]) -> list[dict[str, Any]]:
    return [meta for _row, meta in items]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def dataset_info() -> dict[str, Any]:
    train_columns = {
        "prompt": "instruction",
        "query": "input",
        "chosen": "chosen",
        "rejected": "rejected",
    }
    return {
        "code_dpo_a3_v2_raw_train": {
            "file_name": "code_dpo_a3_v2_raw_train.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": train_columns,
        },
        "code_dpo_a3_v2_clean_train": {
            "file_name": "code_dpo_a3_v2_clean_train.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": train_columns,
        },
        "code_dpo_a3_v2_high_confidence_train": {
            "file_name": "code_dpo_a3_v2_high_confidence_train.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": train_columns,
        },
        "code_dpo_a3_v2_debug_train": {
            "file_name": "code_dpo_a3_v2_debug_train.json",
            "formatting": "alpaca",
            "ranking": True,
            "columns": train_columns,
        },
        "code_dpo_a3_v2_generation_test": {
            "file_name": "code_dpo_a3_v2_generation_test.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return round(float(ordered[lo]), 6)
    frac = pos - lo
    return round(float(ordered[lo] * (1 - frac) + ordered[hi] * frac), 6)


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(float(min(values)), 6),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": round(float(max(values)), 6),
        "mean": round(float(statistics.fmean(values)), 6),
    }


def flag_counts(audits: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for audit in audits:
        for name, value in audit.get("flags", {}).items():
            if value:
                counts[name] += 1
    return dict(sorted(counts.items()))


def drop_counts(audits: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for audit in audits:
        for reason in audit.get("drop_reasons", []):
            counts[reason] += 1
    return dict(sorted(counts.items()))


def prompt_overlap_stats(train_metas: list[dict[str, Any]], test_metas: list[dict[str, Any]]) -> dict[str, Any]:
    train_prompts = {m.get("prompt_sha256_16") for m in train_metas if m.get("prompt_sha256_16")}
    test_prompts = {m.get("prompt_sha256_16") for m in test_metas if m.get("prompt_sha256_16")}
    overlap = sorted(train_prompts & test_prompts)
    return {
        "prompt_overlap_allowed_by_project_policy": True,
        "train_unique_prompt_hashes": len(train_prompts),
        "test_unique_prompt_hashes": len(test_prompts),
        "overlap_unique_prompt_hashes": len(overlap),
        "overlap_examples_sha256_16": overlap[:20],
    }


def generate_quality_report(
    *,
    train_audits: list[dict[str, Any]],
    train_metas: list[dict[str, Any]],
    test_metas: list[dict[str, Any]],
    counts: dict[str, int],
    quality_threshold: float,
    token_length_mode: str,
    chart_files: list[str],
) -> dict[str, Any]:
    scored = [a for a in train_audits if "quality_score" in a]
    metrics = [a.get("metrics", {}) for a in scored]
    scores = [float(a["quality_score"]) for a in scored]
    similarities = [float(m.get("pair_similarity")) for m in metrics if m.get("pair_similarity") is not None]
    chosen_resp = [float(m.get("chosen_response_tokens")) for m in metrics if m.get("chosen_response_tokens") is not None]
    rejected_resp = [float(m.get("rejected_response_tokens")) for m in metrics if m.get("rejected_response_tokens") is not None]
    length_gaps = [float(m.get("chosen_minus_rejected_response_tokens")) for m in metrics if m.get("chosen_minus_rejected_response_tokens") is not None]

    syntax_counts = Counter()
    format_counts = Counter()
    for m in metrics:
        if m.get("chosen_syntax_ok") and m.get("rejected_syntax_ok"):
            syntax_counts["both_syntax_ok"] += 1
        elif m.get("chosen_syntax_ok") and not m.get("rejected_syntax_ok"):
            syntax_counts["chosen_only_syntax_ok"] += 1
        elif (not m.get("chosen_syntax_ok")) and m.get("rejected_syntax_ok"):
            syntax_counts["rejected_only_syntax_ok"] += 1
        else:
            syntax_counts["neither_syntax_ok"] += 1
        if m.get("chosen_has_fence"):
            format_counts["chosen_has_fence"] += 1
        if m.get("rejected_has_fence"):
            format_counts["rejected_has_fence"] += 1
        if m.get("chosen_extractable"):
            format_counts["chosen_extractable"] += 1
        if m.get("rejected_extractable"):
            format_counts["rejected_extractable"] += 1

    return {
        "score_version": SCORE_VERSION,
        "quality_threshold": quality_threshold,
        "score_weights": SCORE_WEIGHTS,
        "token_length_mode": token_length_mode,
        "counts": counts,
        "score_distribution": distribution(scores),
        "similarity_distribution": distribution(similarities),
        "chosen_response_token_distribution": distribution(chosen_resp),
        "rejected_response_token_distribution": distribution(rejected_resp),
        "chosen_minus_rejected_response_token_distribution": distribution(length_gaps),
        "flags": flag_counts(scored),
        "drop_reasons": drop_counts(train_audits),
        "syntax_proxy_counts": dict(sorted(syntax_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "prompt_overlap": prompt_overlap_stats(train_metas, test_metas),
        "charts": chart_files,
        "notes": [
            "Quality score is a static proxy, not a proof of functional correctness.",
            "Prompt overlap is intentionally reported but not filtered for this training-run contract.",
            "Final code correctness should be measured by execution-based evaluation outside A3.",
        ],
    }


def truncate(text: str, limit: int = 300) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_bad_pairs(raw_items: list[tuple[dict[str, str], dict[str, Any]]], limit_per_reason: int = 8) -> dict[str, list[dict[str, Any]]]:
    bad: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, meta in raw_items:
        reasons = list(meta.get("drop_reasons", []))
        if meta.get("quality_score", 1.0) < DEFAULT_QUALITY_THRESHOLD and "below_quality_threshold" not in reasons:
            reasons.append("low_quality_score")
        for reason in reasons:
            if len(bad[reason]) >= limit_per_reason:
                continue
            bad[reason].append(
                {
                    "sample_id": meta.get("sample_id"),
                    "quality_score": meta.get("quality_score"),
                    "flags": meta.get("flags", {}),
                    "instruction_snippet": truncate(row.get("instruction", "")),
                    "chosen_snippet": truncate(row.get("chosen", "")),
                    "rejected_snippet": truncate(row.get("rejected", "")),
                }
            )
    return dict(sorted(bad.items()))


def preview_rows(items: list[tuple[dict[str, str], dict[str, Any]]], n: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, meta in items[:n]:
        preview = {k: truncate(v, 500) for k, v in row.items()}
        preview["_meta"] = {
            "sample_id": meta.get("sample_id"),
            "quality_score": meta.get("quality_score"),
            "views": meta.get("views"),
            "flags": meta.get("flags", {}),
        }
        out.append(preview)
    return out


def make_plots(report_dir: Path, train_audits: list[dict[str, Any]], counts: dict[str, int]) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    scored = [a for a in train_audits if "quality_score" in a]
    metrics = [a.get("metrics", {}) for a in scored]
    chart_files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (report_dir / "plots_skipped.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return []

    def save_hist(values: list[float], title: str, xlabel: str, filename: str, bins: int = 50) -> None:
        if not values:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=bins)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.tight_layout()
        path = report_dir / filename
        plt.savefig(path)
        plt.close()
        chart_files.append(str(path.relative_to(report_dir.parent)))

    save_hist([float(a["quality_score"]) for a in scored], "A3 v2 quality score", "score", "quality_score_hist.png", 40)
    save_hist([float(m.get("pair_similarity", 0.0)) for m in metrics], "Chosen/rejected code-token similarity", "similarity", "pair_similarity_hist.png", 40)
    save_hist([float(m.get("chosen_response_tokens", 0.0)) for m in metrics], "Chosen response token length", "tokens", "chosen_response_tokens_hist.png", 50)
    save_hist([float(m.get("rejected_response_tokens", 0.0)) for m in metrics], "Rejected response token length", "tokens", "rejected_response_tokens_hist.png", 50)
    save_hist([float(m.get("chosen_minus_rejected_response_tokens", 0.0)) for m in metrics], "Chosen - rejected response token length", "token gap", "response_token_gap_hist.png", 50)

    funnel_names = ["raw_train", "clean_train", "high_confidence_train", "debug_train", "generation_test"]
    funnel_values = [counts.get(name, 0) for name in funnel_names]
    plt.figure(figsize=(9, 5))
    plt.bar(funnel_names, funnel_values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("rows")
    plt.title("A3 v2 data-view funnel")
    plt.tight_layout()
    path = report_dir / "filter_funnel.png"
    plt.savefig(path)
    plt.close()
    chart_files.append(str(path.relative_to(report_dir.parent)))
    return chart_files


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    counts = report["counts"]
    prompt_overlap = report["prompt_overlap"]
    lines = [
        "# A3 v2 Preference Quality Report",
        "",
        f"- Score version: `{report['score_version']}`",
        f"- High-confidence threshold: `{report['quality_threshold']}`",
        f"- Token length mode: `{report['token_length_mode']}`",
        f"- Prompt overlap allowed by project policy: `{prompt_overlap['prompt_overlap_allowed_by_project_policy']}`",
        "",
        "## Data-view counts",
        "",
        "| View | Rows |",
        "|---|---:|",
    ]
    for key in ["raw_total", "train_candidates", "test_candidates", "raw_train", "clean_train", "high_confidence_train", "debug_train", "generation_test"]:
        lines.append(f"| `{key}` | {counts.get(key, 0)} |")

    lines.extend(
        [
            "",
            "## Score distribution",
            "",
            "```json",
            json.dumps(report["score_distribution"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Pair-quality flags",
            "",
            "```json",
            json.dumps(report["flags"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Syntax proxy counts",
            "",
            "```json",
            json.dumps(report["syntax_proxy_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Prompt overlap",
            "",
            f"- Train unique prompt hashes: {prompt_overlap['train_unique_prompt_hashes']}",
            f"- Test unique prompt hashes: {prompt_overlap['test_unique_prompt_hashes']}",
            f"- Overlap unique prompt hashes: {prompt_overlap['overlap_unique_prompt_hashes']}",
            "",
            "## Charts",
            "",
        ]
    )
    if report.get("charts"):
        for chart in report["charts"]:
            lines.append(f"- `{chart}`")
    else:
        lines.append("- Plot generation was skipped; see `reports/plots_skipped.txt` if present.")

    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = resolve_project_path(args.source_dir)
    data_root = resolve_project_path(args.data_root)
    output_root = resolve_project_path(args.output_root)
    model_path = resolve_project_path(args.model_path) if args.model_path else None
    run_id = args.run_id or default_run_id()
    data_run_dir = data_root / run_id
    output_run_dir = output_root / run_id

    if data_run_dir.exists():
        raise FileExistsError(f"A3 v2 data run directory already exists: {data_run_dir}")
    data_run_dir.mkdir(parents=True, exist_ok=False)
    (output_run_dir / "logs").mkdir(parents=True, exist_ok=True)

    records, source_files = load_raw_records(source_dir)
    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    raw_test_rows = shuffled[: args.test_size]
    raw_train_rows = shuffled[args.test_size :]
    if args.max_train_samples > 0:
        raw_train_rows = raw_train_rows[: args.max_train_samples]

    estimator = TokenLengthEstimator.create(model_path)
    train_views = build_train_views(
        raw_train_rows,
        prompt_template=args.prompt_template,
        estimator=estimator,
        seed=args.seed,
        quality_threshold=args.quality_threshold,
        debug_size=args.debug_size,
    )
    test_view = build_generation_test(raw_test_rows, prompt_template=args.prompt_template)

    raw_items = train_views["raw_items"]
    clean_items = train_views["clean_items"]
    high_items = train_views["high_items"]
    debug_items = train_views["debug_items"]
    train_audits = train_views["audits"]
    test_items = test_view["items"]
    test_audits = test_view["audits"]

    write_json(data_run_dir / "code_dpo_a3_v2_raw_train.json", rows_only(raw_items))
    write_json(data_run_dir / "code_dpo_a3_v2_clean_train.json", rows_only(clean_items))
    write_json(data_run_dir / "code_dpo_a3_v2_high_confidence_train.json", rows_only(high_items))
    write_json(data_run_dir / "code_dpo_a3_v2_debug_train.json", rows_only(debug_items))
    write_json(data_run_dir / "code_dpo_a3_v2_generation_test.json", rows_only(test_items))
    write_json(data_run_dir / "dataset_info.json", dataset_info())

    train_metas = metas_only(raw_items)
    test_metas = metas_only(test_items)
    write_jsonl(data_run_dir / "metadata_train.jsonl", train_metas)
    write_jsonl(data_run_dir / "metadata_test.jsonl", test_metas)
    write_jsonl(data_run_dir / "pair_audit.jsonl", list(train_audits) + list(test_audits))

    counts = {
        "raw_total": len(records),
        "train_candidates": len(raw_train_rows),
        "test_candidates": len(raw_test_rows),
        "raw_train": len(raw_items),
        "clean_train": len(clean_items),
        "high_confidence_train": len(high_items),
        "debug_train": len(debug_items),
        "generation_test": len(test_items),
        "train_dropped_before_raw": len(raw_train_rows) - len(raw_items),
        "test_dropped_before_generation": len(raw_test_rows) - len(test_items),
    }
    overlap = prompt_overlap_stats(train_metas, test_metas)
    data_statistics = {
        "created_at": utc_now(),
        "run_id": run_id,
        "source_dir": str(source_dir),
        "seed": args.seed,
        "test_size": args.test_size,
        "max_train_samples": args.max_train_samples,
        "quality_threshold": args.quality_threshold,
        "debug_size": args.debug_size,
        "counts": counts,
        "drop_reasons": drop_counts(train_audits + test_audits),
        "prompt_overlap": overlap,
    }

    chart_files = make_plots(data_run_dir / "reports", train_audits, counts) if args.plots else []
    quality_report = generate_quality_report(
        train_audits=train_audits,
        train_metas=train_metas,
        test_metas=test_metas,
        counts=counts,
        quality_threshold=args.quality_threshold,
        token_length_mode=estimator.mode,
        chart_files=chart_files,
    )

    write_json(data_run_dir / "data_statistics.json", data_statistics)
    write_json(data_run_dir / "preference_quality_report.json", quality_report)
    write_markdown_report(data_run_dir / "preference_quality_report.md", quality_report)
    write_json(data_run_dir / "bad_pairs.json", build_bad_pairs(raw_items))
    write_json(
        data_run_dir / "sample_preview.json",
        {
            "raw_train": preview_rows(raw_items),
            "clean_train": preview_rows(clean_items),
            "high_confidence_train": preview_rows(high_items),
            "debug_train": preview_rows(debug_items),
            "generation_test": preview_rows(test_items),
        },
    )
    write_json(data_run_dir / "high_confidence_examples.json", preview_rows(high_items, n=10))

    manifest = {
        "created_at": utc_now(),
        "run_id": run_id,
        "project_root": str(PROJECT_ROOT),
        "source_dir": str(source_dir),
        "source_files": source_files,
        "data_run_dir": str(data_run_dir),
        "output_run_dir": str(output_run_dir),
        "seed": args.seed,
        "test_size": args.test_size,
        "prompt_template": args.prompt_template,
        "score_version": SCORE_VERSION,
        "score_weights": SCORE_WEIGHTS,
        "quality_threshold": args.quality_threshold,
        "debug_size": args.debug_size,
        "model_path_for_token_lengths": str(model_path) if model_path else None,
        "token_length_mode": estimator.mode,
        "qwen3_contract": {"template": "qwen3", "enable_thinking": False},
        "split_policy": {
            "type": "row_level_shuffle_then_head_test",
            "prompt_overlap_allowed_by_project_policy": True,
            "note": "Group split is intentionally not enforced for this practical training contract.",
        },
    }
    write_json(data_run_dir / "run_manifest.json", manifest)

    print(json.dumps({"run_id": run_id, "data_run_dir": str(data_run_dir), "output_run_dir": str(output_run_dir), "counts": counts}, ensure_ascii=False, indent=2))
    return {"run_id": run_id, "data_run_dir": data_run_dir, "output_run_dir": output_run_dir, "counts": counts}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_json_rows(path: Path, expected_columns: set[str], *, min_rows: int = 1) -> int:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise AssertionError(f"{path} is not a JSON list")
    if len(rows) < min_rows:
        raise AssertionError(f"{path} has {len(rows)} rows, expected at least {min_rows}")
    for idx, row in enumerate(rows[:50]):
        if set(row.keys()) != expected_columns:
            raise AssertionError(f"{path} row {idx} columns {sorted(row.keys())} != {sorted(expected_columns)}")
        for key in expected_columns:
            if not isinstance(row[key], str):
                raise AssertionError(f"{path} row {idx} field {key} is not str")
    return len(rows)


def latest_run_id(data_root: Path) -> str:
    if not data_root.exists():
        raise FileNotFoundError(f"No A3 v2 data root exists: {data_root}")
    dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No A3 v2 run directories under: {data_root}")
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def validate_run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_project_path(args.data_root)
    run_id = args.run_id or latest_run_id(data_root)
    data_run_dir = data_root / run_id
    if not data_run_dir.exists():
        raise FileNotFoundError(f"Missing data run directory: {data_run_dir}")

    required = [
        "code_dpo_a3_v2_raw_train.json",
        "code_dpo_a3_v2_clean_train.json",
        "code_dpo_a3_v2_high_confidence_train.json",
        "code_dpo_a3_v2_debug_train.json",
        "code_dpo_a3_v2_generation_test.json",
        "dataset_info.json",
        "run_manifest.json",
        "data_statistics.json",
        "pair_audit.jsonl",
        "metadata_train.jsonl",
        "metadata_test.jsonl",
        "bad_pairs.json",
        "sample_preview.json",
        "preference_quality_report.json",
        "preference_quality_report.md",
    ]
    missing = [name for name in required if not (data_run_dir / name).exists()]
    if missing:
        raise AssertionError(f"Missing required A3 v2 artifacts: {missing}")

    counts = {
        "raw_train": assert_json_rows(data_run_dir / "code_dpo_a3_v2_raw_train.json", TRAIN_COLUMNS),
        "clean_train": assert_json_rows(data_run_dir / "code_dpo_a3_v2_clean_train.json", TRAIN_COLUMNS),
        "high_confidence_train": assert_json_rows(
            data_run_dir / "code_dpo_a3_v2_high_confidence_train.json", TRAIN_COLUMNS, min_rows=args.min_high_confidence
        ),
        "debug_train": assert_json_rows(data_run_dir / "code_dpo_a3_v2_debug_train.json", TRAIN_COLUMNS),
        "generation_test": assert_json_rows(data_run_dir / "code_dpo_a3_v2_generation_test.json", TEST_COLUMNS),
    }

    info = load_json(data_run_dir / "dataset_info.json")
    expected_info = dataset_info()
    for name, spec in expected_info.items():
        if name not in info:
            raise AssertionError(f"dataset_info missing {name}")
        if info[name].get("file_name") != spec["file_name"]:
            raise AssertionError(f"dataset_info {name} file_name mismatch")
        if spec.get("ranking") is True and info[name].get("ranking") is not True:
            raise AssertionError(f"dataset_info {name} ranking should be true")
        if info[name].get("columns") != spec["columns"]:
            raise AssertionError(f"dataset_info {name} columns mismatch")

    stats = load_json(data_run_dir / "data_statistics.json")
    if not stats.get("prompt_overlap", {}).get("prompt_overlap_allowed_by_project_policy"):
        raise AssertionError("prompt overlap policy marker is missing or false")
    report = load_json(data_run_dir / "preference_quality_report.json")
    if report.get("quality_threshold") is None:
        raise AssertionError("quality report missing threshold")
    report_dir = data_run_dir / "reports"
    if not report_dir.exists():
        raise AssertionError("reports directory is missing")

    print(json.dumps({"run_id": run_id, "data_run_dir": str(data_run_dir), "validated_counts": counts}, ensure_ascii=False, indent=2))
    return {"run_id": run_id, "data_run_dir": data_run_dir, "counts": counts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A3 v2 preference data construction and validation")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    prep.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    prep.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prep.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    prep.add_argument("--run_id", type=str, default=os.environ.get("RUN_ID", ""))
    prep.add_argument("--test_size", type=int, default=500)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--max_train_samples", type=int, default=0)
    prep.add_argument("--quality_threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD)
    prep.add_argument("--debug_size", type=int, default=DEFAULT_DEBUG_SIZE)
    prep.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    prep.add_argument("--no_plots", dest="plots", action="store_false", default=True)
    prep.set_defaults(func=prepare_dataset)

    val = sub.add_parser("validate")
    val.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    val.add_argument("--run_id", type=str, default=os.environ.get("RUN_ID", ""))
    val.add_argument("--min_high_confidence", type=int, default=1000)
    val.set_defaults(func=validate_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
