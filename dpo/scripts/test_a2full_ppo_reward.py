#!/usr/bin/env python3
"""Smoke tests for the A2-full PPO reward function."""

from __future__ import annotations

from a2full_ppo_reward import score_message, split_prompt_response


def chat(prompt: str, response: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


def test_good_scores_above_bad() -> None:
    prompt = """
Task:
Write a function to add two numbers.

Your solution must pass these tests:
assert add(1, 2) == 3
assert add(-1, 1) == 0
"""
    good = "def add(a, b):\n    return a + b"
    bad = "def add(a, b):\n    return a - b"
    good_score = score_message(chat(prompt, good)).score
    bad_score = score_message(chat(prompt, bad)).score
    assert good_score > bad_score, (good_score, bad_score)


def test_syntax_penalty() -> None:
    prompt = "Task:\nWrite a function.\nYour solution must pass these tests:\nassert f() == 1"
    broken = "def f(:\n    return 1"
    result = score_message(chat(prompt, broken))
    assert result.syntax_ok is False
    assert result.score < 0.0, result


def test_thinking_tag_penalty() -> None:
    prompt = """
Task:
Write a function to add two numbers.

Your solution must pass these tests:
assert add(1, 2) == 3
"""
    good = "def add(a, b):\n    return a + b"
    tagged = "def add(a, b):\n    return a + b\n</think>\n"
    assert score_message(chat(prompt, good)).score > score_message(chat(prompt, tagged)).score


def test_repetition_and_print_penalty() -> None:
    prompt = """
Task:
Write a function to add two numbers.

Your solution must pass these tests:
assert add(1, 2) == 3
assert add(2, 5) == 7
"""
    compact = "def add(a, b):\n    return a + b"
    repeated = "\n".join(
        [
            "def add(a, b):",
            "    return a + b",
            "print(add(1, 2))",
            "print(add(1, 2))",
            "print(add(1, 2))",
            "print(add(1, 2))",
        ]
    )
    assert score_message(chat(prompt, compact)).score > score_message(chat(prompt, repeated)).score


def test_expected_function_name_penalty() -> None:
    prompt = """
Task:
Write a function to add two numbers.

Your solution must pass these tests:
assert add(1, 2) == 3
"""
    wrong_name = "def solve(a, b):\n    return a + b"
    result = score_message(chat(prompt, wrong_name))
    assert result.components.get("expected_api", 0.0) < 0.0, result


def test_qwen_split() -> None:
    prompt, response = split_prompt_response(chat("hello", "def f():\n    return 1"))
    assert "hello" in prompt
    assert response.startswith("def f()"), response


def main() -> None:
    test_good_scores_above_bad()
    test_syntax_penalty()
    test_thinking_tag_penalty()
    test_repetition_and_print_penalty()
    test_expected_function_name_penalty()
    test_qwen_split()
    print("a2full_ppo_reward smoke tests passed")


if __name__ == "__main__":
    main()
