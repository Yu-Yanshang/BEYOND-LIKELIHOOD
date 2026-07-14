#!/usr/bin/env python3
"""Reward shaping for A2-full PPO code training.

The reward intentionally combines executable signals, static code quality, and
anti-regression penalties. It is clipped and used with PPO KL-to-reference, so
it should nudge the policy toward correct code without rewarding broad drift.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CODE_START_RE = re.compile(r"(?m)^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|@|def\s+\w+|class\s+\w+)")
ASSERT_LINE_RE = re.compile(r"(?m)^\s*assert\s+.+$")
SETUP_IMPORT_LINE_RE = re.compile(r"(?m)^\s*(?:import\s+[A-Za-z_][\w.]*|from\s+[A-Za-z_][\w.]*\s+import\s+[\w*, ]+)\s*$")
REFUSAL_RE = re.compile(r"\b(?:sorry|cannot|can't|unable|as an ai|i do not)\b", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\b(?:todo|pass|notimplemented|not implemented|your code here|placeholder)\b", re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)
CHAT_SPECIAL_RE = re.compile(r"<\|[^|]+?\|>")
EXPLANATION_RE = re.compile(r"(?im)^\s*(?:explanation|reasoning|note|output|example)\s*:")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

PYTHON_KEYWORDS = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "false",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "none",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "true",
    "try",
    "while",
    "with",
    "yield",
}

TASK_STOPWORDS = PYTHON_KEYWORDS | {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "code",
    "correct",
    "for",
    "function",
    "given",
    "in",
    "is",
    "it",
    "of",
    "only",
    "or",
    "pass",
    "python",
    "return",
    "solution",
    "task",
    "that",
    "the",
    "these",
    "this",
    "to",
    "with",
    "write",
    "your",
}

SAFE_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "copy",
    "datetime",
    "decimal",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "typing",
}

UNSAFE_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}


@dataclasses.dataclass
class RewardResult:
    score: float
    components: dict[str, float]
    prompt: str
    response: str
    code: str
    tests_found: int
    tests_passed: int
    syntax_ok: bool
    safe_to_execute: bool
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["prompt"] = self.prompt[:800]
        data["response"] = self.response[:800]
        data["code"] = self.code[:800]
        return data


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def clamp(value: float, low: float = -3.0, high: float = 3.0) -> float:
    return max(low, min(high, value))


def split_prompt_response(message: str) -> tuple[str, str]:
    text = normalize_text(message)
    markers = [
        "<|im_start|>assistant",
        "<|assistant|>",
        "\nassistant\n",
        "\nAssistant:",
        "\nASSISTANT:",
    ]
    for marker in markers:
        if marker in text:
            prompt, response = text.rsplit(marker, 1)
            response = response.lstrip("\n :")
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>", 1)[0]
            return prompt, normalize_text(response)

    if "[BEGIN]" in text:
        prompt, response = text.rsplit("[BEGIN]", 1)
        response = response.split("[DONE]", 1)[0]
        return prompt, normalize_text(response)

    return "", text


def extract_code(text: str) -> str:
    text = normalize_text(text)
    if "[BEGIN]" in text:
        text = text.rsplit("[BEGIN]", 1)[-1]
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]

    fenced = FENCED_CODE_RE.findall(text)
    if fenced:
        return normalize_text(fenced[-1])

    match = CODE_START_RE.search(text)
    if match:
        return normalize_text(text[match.start() :])

    return text


def preamble_text(response: str) -> str:
    response = normalize_text(response)
    if response.startswith("[BEGIN]"):
        return ""
    if FENCED_CODE_RE.search(response):
        return ""
    match = CODE_START_RE.search(response)
    if not match:
        return response
    return normalize_text(response[: match.start()])


def parse_code(code: str) -> tuple[bool, ast.AST | None]:
    try:
        return True, ast.parse(code)
    except SyntaxError:
        return False, None


def extract_prompt_tests(prompt: str, limit: int = 8) -> list[str]:
    tests = []
    for match in ASSERT_LINE_RE.finditer(prompt):
        line = match.group(0).strip()
        if line not in tests:
            tests.append(line)
        if len(tests) >= limit:
            break
    return tests


def expected_call_names(tests: list[str]) -> set[str]:
    names: set[str] = set()
    for test in tests:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in {"list", "dict", "set", "tuple", "len", "sum", "sorted", "round"}:
                    names.add(node.func.id)
    return names


def defined_callable_names(tree: ast.AST | None) -> set[str]:
    if not isinstance(tree, ast.Module):
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def extract_prompt_setup(prompt: str, limit: int = 6) -> list[str]:
    setup = []
    for match in SETUP_IMPORT_LINE_RE.finditer(prompt):
        line = match.group(0).strip()
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        safe, _ = safety_check(tree)
        if safe and line not in setup:
            setup.append(line)
        if len(setup) >= limit:
            break
    return setup


def safe_import_root(name: str) -> bool:
    return name.split(".", 1)[0] in SAFE_IMPORT_ROOTS


def safety_check(tree: ast.AST | None) -> tuple[bool, list[str]]:
    if tree is None:
        return False, ["syntax"]

    reasons: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not safe_import_root(alias.name):
                    reasons.append(f"import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not safe_import_root(module):
                reasons.append(f"from:{module}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in UNSAFE_NAMES:
                reasons.append(f"call:{func.id}")
            elif isinstance(func, ast.Attribute):
                root = func.value.id if isinstance(func.value, ast.Name) else ""
                if root in {"os", "sys", "subprocess", "socket", "shutil", "pathlib"}:
                    reasons.append(f"attr:{root}.{func.attr}")
        elif isinstance(node, (ast.Delete, ast.Global, ast.Nonlocal)):
            reasons.append(type(node).__name__.lower())

    return not reasons, reasons[:6]


def top_level_print_or_input(tree: ast.AST | None) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for node in tree.body:
        target = node.value if isinstance(node, ast.Expr) else node
        if isinstance(target, ast.Call):
            if isinstance(target.func, ast.Name) and target.func.id in {"print", "input"}:
                return True
    return False


def top_level_io_count(tree: ast.AST | None) -> int:
    if not isinstance(tree, ast.Module):
        return 0
    count = 0
    for node in tree.body:
        target = node.value if isinstance(node, ast.Expr) else node
        if isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id in {"print", "input"}:
            count += 1
    return count


def has_callable(tree: ast.AST | None) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body)


def code_identifier_tokens(tree: ast.AST | None, code: str) -> set[str]:
    tokens: set[str] = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                tokens.add(node.id.lower())
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                tokens.update(part.lower() for part in node.name.split("_") if part)

    if not tokens:
        tokens.update(token.lower() for token in IDENT_RE.findall(code))
    return {token for token in tokens if token not in PYTHON_KEYWORDS and len(token) > 2}


def prompt_keywords(prompt: str) -> set[str]:
    clean = re.sub(r"(?ms)Your solution must pass these tests:.*", " ", prompt)
    return {
        token.lower()
        for token in IDENT_RE.findall(clean)
        if token.lower() not in TASK_STOPWORDS and len(token) > 2
    }


def relevance_score(prompt: str, tree: ast.AST | None, code: str) -> float:
    prompt_terms = prompt_keywords(prompt)
    if not prompt_terms:
        return 0.0
    code_terms = code_identifier_tokens(tree, code)
    overlap = len(prompt_terms & code_terms)
    return min(0.45, 0.09 * overlap)


def repetition_component(code: str) -> float:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    if len(lines) < 8:
        return 0.0

    counts = Counter(lines)
    repeated_lines = sum(count - 1 for line, count in counts.items() if count > 1 and not line.startswith("#"))
    penalty = min(1.0, repeated_lines * 0.12)

    tokens = IDENT_RE.findall(code)
    if len(tokens) >= 24:
        trigrams = [" ".join(tokens[i : i + 3]).lower() for i in range(len(tokens) - 2)]
        trigram_counts = Counter(trigrams)
        repeated_trigrams = sum(count - 1 for count in trigram_counts.values() if count > 2)
        penalty += min(0.6, repeated_trigrams * 0.03)

    return -min(1.4, penalty) if penalty else 0.0


def length_component(code: str) -> float:
    words = code.split()
    if len(words) < 8:
        return -0.45
    if len(code) > 2600:
        return -1.0
    if len(code) > 1800:
        return -0.65
    if len(code) > 1200:
        return -0.3
    if 20 <= len(words) <= 180:
        return 0.2
    if len(words) > 260:
        return -0.35
    return 0.0


def expected_api_component(tree: ast.AST | None, tests: list[str]) -> float:
    expected = expected_call_names(tests)
    if not expected:
        return 0.0
    defined = defined_callable_names(tree)
    missing = expected - defined
    if not missing:
        return 0.45
    return -min(1.0, 0.35 * len(missing))


def exact_test_echo_count(code: str, tests: list[str]) -> int:
    normalized_code = "\n".join(line.strip() for line in code.splitlines())
    return sum(1 for test in tests if test.strip() and test.strip() in normalized_code)


def limit_resources(memory_mb: int, timeout: float) -> None:
    try:
        import resource

        memory_bytes = memory_mb * 1024 * 1024
        cpu_seconds = max(1, int(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    except Exception:
        return


def run_prompt_tests(
    code: str,
    tests: list[str],
    setup: list[str],
    timeout: float,
    memory_mb: int,
) -> tuple[int, str]:
    if not tests:
        return 0, ""

    runner = "\n\n".join(
        [
            "import faulthandler\nfaulthandler.enable()",
            "\n".join(setup),
            code,
            "\n".join(tests),
        ]
    )

    with tempfile.TemporaryDirectory(prefix="a2full_reward_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["HOME"] = tmpdir
        env["PYTHONPATH"] = ""
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = lambda: limit_resources(memory_mb, timeout)
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                **popen_kwargs,
            )
        except subprocess.TimeoutExpired:
            return 0, "timeout"

    if result.returncode == 0:
        return len(tests), ""
    return 0, "runtime_error"


def score_message(
    message: str,
    *,
    enable_prompt_tests: bool = True,
    test_timeout: float = 1.5,
    memory_mb: int = 512,
) -> RewardResult:
    prompt, response = split_prompt_response(message)
    code = extract_code(response)
    syntax_ok, tree = parse_code(code)
    safe, safety_reasons = safety_check(tree)
    tests = extract_prompt_tests(prompt)
    setup = extract_prompt_setup(prompt)

    components: dict[str, float] = {}
    score = 0.0

    if not response or not code:
        return RewardResult(-3.0, {"empty": -3.0}, prompt, response, code, 0, 0, False, False, "empty")

    if REFUSAL_RE.search(response):
        components["refusal"] = -1.0
    if PLACEHOLDER_RE.search(code):
        components["placeholder"] = -0.6
    if THINK_TAG_RE.search(response):
        components["thinking_tag"] = -1.4
    if CHAT_SPECIAL_RE.search(response):
        components["chat_special_token"] = -0.7
    if FENCED_CODE_RE.search(response):
        components["markdown_fence"] = -0.3
    if EXPLANATION_RE.search(response):
        components["explanation_text"] = -0.35
    if preamble_text(response):
        components["preamble"] = -0.35

    components["syntax"] = 0.85 if syntax_ok else -1.35
    components["safety"] = 0.25 if safe else -0.75
    components["callable"] = 0.3 if has_callable(tree) else -0.2
    io_count = top_level_io_count(tree)
    components["top_level_io"] = -min(1.0, 0.35 * io_count) if io_count else 0.0
    components["length"] = length_component(code)
    components["relevance"] = relevance_score(prompt, tree, code)
    components["expected_api"] = expected_api_component(tree, tests)
    components["repetition"] = repetition_component(code)

    tests_passed = 0
    error_type = ""
    if tests:
        if enable_prompt_tests and syntax_ok and safe:
            tests_passed, error_type = run_prompt_tests(code, tests, setup, test_timeout, memory_mb)
            pass_rate = tests_passed / len(tests)
            components["prompt_tests"] = -0.7 + 2.3 * pass_rate
            if tests_passed == len(tests):
                components["all_tests"] = 0.35
        elif not safe:
            error_type = ",".join(safety_reasons) or "unsafe"
            components["prompt_tests"] = -0.5
        elif not syntax_ok:
            error_type = "syntax"
            components["prompt_tests"] = -0.5
    else:
        components["no_prompt_tests"] = -0.1

    echoed_tests = exact_test_echo_count(code, tests)
    if echoed_tests:
        components["echoed_tests"] = -min(1.2, 0.35 * echoed_tests)
    elif "assert " in code and tests:
        components["assert_in_answer"] = -0.35

    score = clamp(sum(components.values()))
    return RewardResult(
        score=score,
        components={key: round(value, 4) for key, value in components.items() if value},
        prompt=prompt,
        response=response,
        code=code,
        tests_found=len(tests),
        tests_passed=tests_passed,
        syntax_ok=syntax_ok,
        safe_to_execute=safe,
        error_type=error_type,
    )


def rank_score(text: str) -> float:
    return score_message(text).score


def main() -> None:
    message = sys.stdin.read()
    result = score_message(message)
    print(result.to_dict())


if __name__ == "__main__":
    main()
