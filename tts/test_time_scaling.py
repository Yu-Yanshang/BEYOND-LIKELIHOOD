#!/usr/bin/env python3
"""Test-time scaling strategies for MBPP-style Python code generation.

Implements greedy decoding, CoT, Self-Consistency, Best-of-N, Reflexion, and
Tree of Thoughts while reusing the existing sandboxed MBPP execution helpers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from cot_code_example import build_chat_messages, extract_final_code, normalize_text
    from evaluate_cot_code import (
        load_rows,
        row_to_reference,
        row_to_task,
        row_to_tests,
        save_json,
        save_jsonl,
        score_case,
    )
except ImportError:
    from tts.cot_code_example import build_chat_messages, extract_final_code, normalize_text
    from tts.evaluate_cot_code import (
        load_rows,
        row_to_reference,
        row_to_task,
        row_to_tests,
        save_json,
        save_jsonl,
        score_case,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "best_path" / "runs" / "best_full_dpo_a5_20260702" / "models" / "a2_full"
DEFAULT_INPUT = PROJECT_ROOT.parent / "mbpp" / "sanitized" / "test-00000-of-00001.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "tts" / "outputs" / "test_time_scaling_a2_full"
ALL_STRATEGIES = ("greedy", "cot", "self_consistency", "best_of_n", "reflexion", "tot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate inference-time scaling strategies on Python code tasks."
    )
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strategy",
        choices=(*ALL_STRATEGIES, "compare", "all"),
        default="compare",
        help="compare/all runs every strategy and writes a comparison report.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 uses every input row.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--num_samples", type=int, default=5, help="Self-Consistency samples.")
    parser.add_argument("--best_of_n", type=int, default=5, help="Best-of-N candidate count.")
    parser.add_argument("--reflexion_rounds", type=int, default=2)
    parser.add_argument("--tree_width", type=int, default=3)
    parser.add_argument("--tree_depth", type=int, default=2)
    parser.add_argument("--test_timeout", type=float, default=5.0)
    parser.add_argument("--memory_mb", type=int, default=1024)
    args = parser.parse_args()

    positive = {
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": args.num_samples,
        "best_of_n": args.best_of_n,
        "tree_width": args.tree_width,
        "tree_depth": args.tree_depth,
        "memory_mb": args.memory_mb,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"--{name} must be at least 1")
    if args.reflexion_rounds < 0:
        parser.error("--reflexion_rounds must be at least 0")
    if not 0.0 <= args.temperature:
        parser.error("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top_p must be in (0, 1]")
    return args


@dataclass
class Usage:
    generated_sequences: int = 0
    generated_tokens: int = 0
    generation_calls: int = 0

    def snapshot(self) -> tuple[int, int, int]:
        return self.generated_sequences, self.generated_tokens, self.generation_calls

    def delta(self, before: tuple[int, int, int]) -> dict[str, int]:
        return {
            "generated_sequences": self.generated_sequences - before[0],
            "generated_tokens": self.generated_tokens - before[1],
            "generation_calls": self.generation_calls - before[2],
        }


class ModelGenerator:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not args.model_path.exists():
            raise FileNotFoundError(f"Missing model directory: {args.model_path}")
        self.args = args
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        adapter_config = args.model_path / "adapter_config.json"
        if adapter_config.is_file():
            # Loading an adapter directly through Transformers delegates to its
            # PEFT integration.  Some installed Transformers/PEFT combinations
            # use incompatible private APIs, so load the base model and adapter
            # explicitly through PEFT instead.
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(args.model_path)
            base_model_path = peft_config.base_model_name_or_path
            if not base_model_path:
                raise ValueError(f"Adapter has no base_model_name_or_path: {args.model_path}")
            print(
                f"Loading PEFT adapter {args.model_path} on base model {base_model_path}",
                flush=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                device_map=args.device_map,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, args.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype="auto",
                device_map=args.device_map,
                trust_remote_code=True,
            )
        self.model.eval()
        self.usage = Usage()

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return getattr(self.model, "device", "cpu")

    def _format(self, messages: list[dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        joined = "\n\n".join(f"{item['role'].title()}:\n{item['content']}" for item in messages)
        return f"{joined}\n\nAssistant:\n"

    def generate(
        self,
        requests: list[list[dict[str, str]]],
        *,
        sample: bool,
        max_new_tokens: int | None = None,
        label: str = "generation",
    ) -> list[str]:
        outputs: list[str] = []
        max_tokens = max_new_tokens or self.args.max_new_tokens
        for start in range(0, len(requests), self.args.batch_size):
            batch = requests[start : start + self.args.batch_size]
            prompts = [self._format(messages) for messages in batch]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_input_tokens,
            ).to(self._device())
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": sample,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            if sample:
                kwargs.update(temperature=max(self.args.temperature, 1e-5), top_p=self.args.top_p)
            else:
                # Qwen's saved generation config contains sampling controls.
                # Explicitly clear them for deterministic decoding so recent
                # Transformers versions do not emit irrelevant warnings.
                kwargs.update(temperature=None, top_p=None, top_k=None)
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **kwargs)
            response_ids = generated[:, inputs["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            outputs.extend(normalize_text(text) for text in texts)
            self.usage.generated_sequences += len(batch)
            self.usage.generated_tokens += int(
                (response_ids != (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)).sum().item()
            )
            self.usage.generation_calls += 1
            print(f"{label}: {min(start + len(batch), len(requests))}/{len(requests)}", flush=True)
        return outputs


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(load_rows(args.input_file)):
        task = row_to_task(raw)
        if not task:
            continue
        setup, tests = row_to_tests(raw)
        rows.append(
            {
                "index": index,
                "task": task,
                "reference": normalize_text(row_to_reference(raw)),
                "test_setup": setup,
                "tests": tests,
                "raw": raw,
            }
        )
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No valid tasks found in {args.input_file}")
    return rows


def greedy_messages(task: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a Python programming assistant. Return one complete executable solution "
                "in a single ```python code block. Do not include analysis outside the code block."
            ),
        },
        {"role": "user", "content": f"Task:\n{task}"},
    ]


def append_instruction(messages: list[dict[str, str]], instruction: str) -> list[dict[str, str]]:
    result = [dict(item) for item in messages]
    result[-1]["content"] = f"{result[-1]['content']}\n\n{instruction}"
    return result


def canonical_code(code: str) -> str:
    try:
        return ast.unparse(ast.parse(code)).strip()
    except (SyntaxError, ValueError):
        return re.sub(r"\s+", " ", code).strip()


def format_score(response: str, code: str) -> float:
    score = 0.0
    lower = response.lower()
    if code.strip():
        score += 0.5
    if "```" in response:
        score += 0.25
    if "final code" in lower:
        score += 0.25
    if len(code) > 8000:
        score -= 0.5
    return score


def verifier_score(case: dict[str, Any]) -> float:
    total_tests = int(case["total_tests"])
    test_ratio = case["passed_tests"] / total_tests if total_tests else 0.0
    return (
        2.0 * float(case["syntax_ok"])
        + 6.0 * test_ratio
        + 3.0 * float(case["passed"])
        + format_score(case["response"], case["final_code"])
    )


def evaluate_candidate(
    row: dict[str, Any], response: str, args: argparse.Namespace, candidate_id: int
) -> dict[str, Any]:
    case = score_case(row, response, args)
    case["candidate_id"] = candidate_id
    case["verifier_score"] = verifier_score(case)
    return case


def compact_candidate(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": case.get("candidate_id"),
        "response": case["response"],
        "final_code": case["final_code"],
        "syntax_ok": case["syntax_ok"],
        "passed": case["passed"],
        "passed_tests": case["passed_tests"],
        "total_tests": case["total_tests"],
        "verifier_score": case["verifier_score"],
        "test_results": case["test_results"],
    }


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            item["verifier_score"],
            item["passed_tests"],
            int(item["syntax_ok"]),
            -len(item["final_code"]),
            -int(item.get("candidate_id", 0)),
        ),
    )


def final_case(selected: dict[str, Any], strategy: str, **details: Any) -> dict[str, Any]:
    result = dict(selected)
    result["strategy"] = strategy
    result.update(details)
    return result


def run_single_path(
    strategy: str,
    rows: list[dict[str, Any]],
    generator: ModelGenerator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if strategy == "greedy":
        requests = [greedy_messages(row["task"]) for row in rows]
    else:
        requests = [build_chat_messages(row["task"]) for row in rows]
    responses = generator.generate(requests, sample=False, label=strategy)
    cases = [
        final_case(evaluate_candidate(row, response, args, 0), strategy)
        for row, response in zip(rows, responses, strict=True)
    ]
    return cases, {"candidates_per_task": 1}


def run_self_consistency(
    rows: list[dict[str, Any]], generator: ModelGenerator, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests: list[list[dict[str, str]]] = []
    owners: list[tuple[int, int]] = []
    for row_pos, row in enumerate(rows):
        for sample_id in range(args.num_samples):
            requests.append(
                append_instruction(
                    build_chat_messages(row["task"]),
                    f"Use independent reasoning path {sample_id + 1}; do not copy a stock solution.",
                )
            )
            owners.append((row_pos, sample_id))
    responses = generator.generate(requests, sample=True, label="self_consistency")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for response, (row_pos, sample_id) in zip(responses, owners, strict=True):
        grouped[row_pos].append(evaluate_candidate(rows[row_pos], response, args, sample_id))

    final: list[dict[str, Any]] = []
    consistency_values: list[float] = []
    for row_pos, row in enumerate(rows):
        candidates = grouped[row_pos]
        by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_code[canonical_code(candidate["final_code"])].append(candidate)
        winning_group = max(
            by_code.values(), key=lambda group: (len(group), select_best(group)["verifier_score"])
        )
        selected = select_best(winning_group)
        consistency = len(winning_group) / len(candidates)
        consistency_values.append(consistency)
        final.append(
            final_case(
                selected,
                "self_consistency",
                selection="majority_canonical_code_then_verifier",
                majority_count=len(winning_group),
                majority_consistency=consistency,
                unique_answers=len(by_code),
                candidates=[compact_candidate(item) for item in candidates],
            )
        )
    return final, {
        "num_samples": args.num_samples,
        "avg_majority_consistency": sum(consistency_values) / len(consistency_values),
        "unanimous_rate": sum(value == 1.0 for value in consistency_values) / len(consistency_values),
    }


def run_best_of_n(
    rows: list[dict[str, Any]], generator: ModelGenerator, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests: list[list[dict[str, str]]] = []
    owners: list[tuple[int, int]] = []
    for row_pos, row in enumerate(rows):
        for candidate_id in range(args.best_of_n):
            requests.append(
                append_instruction(
                    build_chat_messages(row["task"]),
                    f"Produce diverse candidate {candidate_id + 1}. Check edge cases before finalizing.",
                )
            )
            owners.append((row_pos, candidate_id))
    responses = generator.generate(requests, sample=True, label="best_of_n")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for response, (row_pos, candidate_id) in zip(responses, owners, strict=True):
        grouped[row_pos].append(evaluate_candidate(rows[row_pos], response, args, candidate_id))

    cases: list[dict[str, Any]] = []
    for row_pos in range(len(rows)):
        candidates = grouped[row_pos]
        selected = select_best(candidates)
        cases.append(
            final_case(
                selected,
                "best_of_n",
                selection="syntax_plus_tests_plus_format_verifier",
                candidates=[compact_candidate(item) for item in candidates],
            )
        )
    return cases, {"best_of_n": args.best_of_n, "verifier_max_score": 12.0}


def feedback_from_case(case: dict[str, Any]) -> str:
    if not case["syntax_ok"]:
        try:
            ast.parse(case["final_code"])
        except SyntaxError as exc:
            return f"Syntax error: {exc.msg} at line {exc.lineno}."
        return "The candidate is not valid Python syntax."
    failures = [item for item in case["test_results"] if not item["passed"]]
    if not failures:
        return "No public test failed. Recheck edge cases and simplify the implementation."
    messages = []
    for item in failures[:3]:
        detail = normalize_text(item.get("stderr") or item.get("stdout") or item.get("error_type"))
        messages.append(detail[-1200:] if detail else str(item.get("error_type", "test failure")))
    return "\n---\n".join(messages)


def reflexion_messages(row: dict[str, Any], previous: dict[str, Any], round_id: int) -> list[dict[str, str]]:
    feedback = feedback_from_case(previous)
    content = f"""Task:
{row['task']}

Previous candidate:
```python
{previous['final_code'][-8000:]}
```

Verifier feedback:
{feedback}

Reflexion round {round_id}:
1. Explain the root cause briefly.
2. State the correction.
3. Return the complete corrected program under Final code.
Do not return a patch or only a fragment."""
    return [
        {
            "role": "system",
            "content": (
                "You are a Python debugging expert. Use execution feedback to correct the solution. "
                "Answer with Reflection, Fix, and Final code sections."
            ),
        },
        {"role": "user", "content": content},
    ]


def run_reflexion(
    rows: list[dict[str, Any]], generator: ModelGenerator, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_responses = generator.generate(
        [build_chat_messages(row["task"]) for row in rows], sample=False, label="reflexion_initial"
    )
    histories: list[list[dict[str, Any]]] = [
        [evaluate_candidate(row, response, args, 0)]
        for row, response in zip(rows, initial_responses, strict=True)
    ]
    for round_id in range(1, args.reflexion_rounds + 1):
        pending = [index for index, history in enumerate(histories) if not history[-1]["passed"]]
        if not pending:
            break
        requests = [reflexion_messages(rows[index], histories[index][-1], round_id) for index in pending]
        responses = generator.generate(requests, sample=False, label=f"reflexion_round_{round_id}")
        for index, response in zip(pending, responses, strict=True):
            histories[index].append(evaluate_candidate(rows[index], response, args, round_id))

    cases: list[dict[str, Any]] = []
    corrected = 0
    degraded = 0
    for history in histories:
        initial = history[0]
        selected = select_best(history)
        corrected += int(not initial["passed"] and selected["passed"])
        degraded += int(initial["passed"] and not selected["passed"])
        cases.append(
            final_case(
                selected,
                "reflexion",
                selection="best_verifier_score_across_refinement_rounds",
                initial_passed=initial["passed"],
                rounds_attempted=len(history) - 1,
                reflection_history=[compact_candidate(item) for item in history],
            )
        )
    initial_pass_rate = sum(history[0]["passed"] for history in histories) / len(histories)
    final_pass_rate = sum(case["passed"] for case in cases) / len(cases)
    return cases, {
        "reflexion_rounds": args.reflexion_rounds,
        "initial_pass_rate": initial_pass_rate,
        "final_pass_rate": final_pass_rate,
        "corrected_tasks": corrected,
        "degraded_tasks": degraded,
    }


def thought_messages(task: str, parent: str, depth: int, branch: int) -> list[dict[str, str]]:
    parent_text = parent if parent else "No prior plan; start from the task."
    return [
        {
            "role": "system",
            "content": (
                "You are planning a Python solution. Return only an Approach section, not code. "
                "The approach must identify the algorithm, required function signature, edge cases, and complexity."
            ),
        },
        {
            "role": "user",
            "content": f"""Task:
{task}

Current path:
{parent_text}

Expand search depth {depth} with alternative branch {branch}. Improve or replace weak choices.
Approach:""",
        },
    ]


def score_thought(text: str, task: str, depth: int) -> float:
    lower = text.lower()
    score = min(len(text.split()) / 80.0, 1.0)
    score += 0.35 * sum(term in lower for term in ("edge", "complex", "return", "function", "test"))
    task_names = re.findall(r"\b[a-zA-Z_][a-zA-Z_0-9]*\b", task)
    score += 0.15 * sum(name.lower() in lower for name in task_names[:8])
    score += 0.1 * depth
    if "```" in text or re.search(r"\bdef\s+\w+", text):
        score -= 1.0
    if len(text) > 5000:
        score -= 0.5
    return score


def code_from_thought_messages(task: str, thought: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a Python programming assistant. Follow the selected plan and answer with "
                "Reasoning, Key steps, and one complete executable solution under Final code."
            ),
        },
        {"role": "user", "content": f"Task:\n{task}\n\nSelected reasoning path:\n{thought}"},
    ]


def run_tot(
    rows: list[dict[str, Any]], generator: ModelGenerator, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    beams: dict[int, list[dict[str, Any]]] = {
        index: [{"text": "", "thought_score": 0.0, "depth": 0, "parent": None}]
        for index in range(len(rows))
    }
    expanded_nodes = 0
    for depth in range(1, args.tree_depth + 1):
        requests: list[list[dict[str, str]]] = []
        owners: list[tuple[int, dict[str, Any], int]] = []
        for row_pos, row in enumerate(rows):
            for parent in beams[row_pos]:
                for branch in range(1, args.tree_width + 1):
                    requests.append(thought_messages(row["task"], parent["text"], depth, branch))
                    owners.append((row_pos, parent, branch))
        outputs = generator.generate(
            requests,
            sample=True,
            max_new_tokens=min(args.max_new_tokens, 384),
            label=f"tot_depth_{depth}",
        )
        expanded_nodes += len(outputs)
        children: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for output, (row_pos, parent, branch) in zip(outputs, owners, strict=True):
            combined = normalize_text(
                f"{parent['text']}\n\nDepth {depth}, branch {branch}:\n{output}" if parent["text"] else output
            )
            children[row_pos].append(
                {
                    "text": combined,
                    "thought_score": parent["thought_score"]
                    + score_thought(output, rows[row_pos]["task"], depth),
                    "depth": depth,
                    "branch": branch,
                    "parent_score": parent["thought_score"],
                }
            )
        for row_pos in range(len(rows)):
            beams[row_pos] = sorted(
                children[row_pos],
                key=lambda item: item["thought_score"],
                reverse=True,
            )[: args.tree_width]

    requests = []
    owners: list[tuple[int, int]] = []
    for row_pos, row in enumerate(rows):
        for path_id, thought in enumerate(beams[row_pos]):
            requests.append(code_from_thought_messages(row["task"], thought["text"]))
            owners.append((row_pos, path_id))
    responses = generator.generate(requests, sample=False, label="tot_final_code")
    candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for response, (row_pos, path_id) in zip(responses, owners, strict=True):
        candidate = evaluate_candidate(rows[row_pos], response, args, path_id)
        candidate["thought"] = beams[row_pos][path_id]
        candidates[row_pos].append(candidate)

    cases: list[dict[str, Any]] = []
    for row_pos in range(len(rows)):
        selected = select_best(candidates[row_pos])
        cases.append(
            final_case(
                selected,
                "tot",
                selection="beam_thought_search_then_code_verifier",
                selected_thought=selected["thought"],
                search_paths=[
                    {"thought": item["thought"], "candidate": compact_candidate(item)}
                    for item in candidates[row_pos]
                ],
            )
        )
    return cases, {
        "tree_width": args.tree_width,
        "tree_depth": args.tree_depth,
        "expanded_thought_nodes": expanded_nodes,
        "final_code_paths": len(rows) * args.tree_width,
    }


def summarize(
    strategy: str,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    usage: dict[str, int],
    elapsed: float,
    extras: dict[str, Any],
    peak_gpu_mb: float,
) -> dict[str, Any]:
    total = len(cases)
    with_tests = [case for case in cases if case["total_tests"] > 0]
    passed_tasks = sum(case["passed"] for case in with_tests)
    total_tests = sum(case["total_tests"] for case in with_tests)
    passed_tests = sum(case["passed_tests"] for case in with_tests)
    metrics: dict[str, Any] = {
        "strategy": strategy,
        "model_path": str(args.model_path),
        "input_file": str(args.input_file),
        "total": total,
        "syntax_pass_rate": sum(case["syntax_ok"] for case in cases) / total,
        "pass_at_1": passed_tasks / len(with_tests) if with_tests else None,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else None,
        "passed_tasks": passed_tasks if with_tests else None,
        "total_tests": total_tests if with_tests else None,
        "passed_tests": passed_tests if with_tests else None,
        "elapsed_seconds": elapsed,
        "seconds_per_task": elapsed / total,
        "peak_gpu_memory_mb": peak_gpu_mb,
        **usage,
        "avg_generated_sequences_per_task": usage["generated_sequences"] / total,
        "avg_generated_tokens_per_task": usage["generated_tokens"] / total,
        "seed": args.seed,
        **extras,
    }
    return metrics


def run_strategy(
    strategy: str,
    rows: list[dict[str, Any]],
    generator: ModelGenerator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = generator.torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = generator.usage.snapshot()
    started = time.perf_counter()
    if strategy in ("greedy", "cot"):
        cases, extras = run_single_path(strategy, rows, generator, args)
    elif strategy == "self_consistency":
        cases, extras = run_self_consistency(rows, generator, args)
    elif strategy == "best_of_n":
        cases, extras = run_best_of_n(rows, generator, args)
    elif strategy == "reflexion":
        cases, extras = run_reflexion(rows, generator, args)
    elif strategy == "tot":
        cases, extras = run_tot(rows, generator, args)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    elapsed = time.perf_counter() - started
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    )
    usage = generator.usage.delta(before)
    metrics = summarize(strategy, cases, args, usage, elapsed, extras, peak_gpu_mb)
    return cases, metrics


def write_comparison(output_dir: Path, metrics: list[dict[str, Any]]) -> None:
    save_json(output_dir / "comparison.json", metrics)
    fields = [
        "strategy",
        "total",
        "syntax_pass_rate",
        "pass_at_1",
        "avg_test_pass_rate",
        "elapsed_seconds",
        "seconds_per_task",
        "generated_sequences",
        "generated_tokens",
        "avg_generated_tokens_per_task",
        "peak_gpu_memory_mb",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    lines = [
        "# Test-Time Scaling Comparison",
        "",
        "| Strategy | Syntax pass | pass@1 | Avg test pass | Seconds/task | Sequences | Tokens/task |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics:
        values = {
            key: ("n/a" if item.get(key) is None else f"{item[key]:.4f}")
            for key in ("syntax_pass_rate", "pass_at_1", "avg_test_pass_rate", "seconds_per_task")
        }
        lines.append(
            f"| {item['strategy']} | {values['syntax_pass_rate']} | {values['pass_at_1']} | "
            f"{values['avg_test_pass_rate']} | {values['seconds_per_task']} | "
            f"{item['generated_sequences']} | {item['avg_generated_tokens_per_task']:.1f} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rows = build_rows(args)
    strategies = ALL_STRATEGIES if args.strategy in ("compare", "all") else (args.strategy,)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_file}; tasks: {len(rows)}; strategies: {', '.join(strategies)}")

    generator = ModelGenerator(args)
    try:
        generator.torch.manual_seed(args.seed)
        if generator.torch.cuda.is_available():
            generator.torch.cuda.manual_seed_all(args.seed)
        all_metrics: list[dict[str, Any]] = []
        for strategy in strategies:
            cases, metrics = run_strategy(strategy, rows, generator, args)
            strategy_dir = args.output_dir / strategy
            save_json(strategy_dir / "metrics.json", metrics)
            save_jsonl(strategy_dir / "cases.jsonl", cases)
            all_metrics.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        write_comparison(args.output_dir, all_metrics)
        print(f"Wrote reports to {args.output_dir}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()
