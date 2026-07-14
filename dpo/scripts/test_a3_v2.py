#!/usr/bin/env python3
"""Small unit tests for A3 v2 helpers. Does not touch project data outputs."""

from __future__ import annotations

import random
from pathlib import Path

from dpo.a3_v2 import pipeline as p


def assert_true(expr: bool, msg: str) -> None:
    if not expr:
        raise AssertionError(msg)


def test_extract_and_syntax() -> None:
    fenced = "Here is code:\n```python\ndef add(a, b):\n    return a + b\n```"
    extracted = p.extract_code(fenced)
    assert_true(extracted["has_fence"], "fenced block not detected")
    ok, _ = p.syntax_ok(extracted["code"])
    assert_true(ok, "valid fenced Python should parse")

    bad = "```python\ndef broken(:\n    pass\n```"
    extracted_bad = p.extract_code(bad)
    ok, _ = p.syntax_ok(extracted_bad["code"])
    assert_true(not ok, "invalid Python should fail syntax check")


def test_score_flags() -> None:
    estimator = p.TokenLengthEstimator.create(Path("/definitely/missing/qwen3"))
    instruction = "Task: write add"
    chosen = "```python\ndef add(a, b):\n    return a + b\n```"
    rejected = "```python\ndef add(a, b):\n    return a - b\n```"
    score = p.score_pair(instruction, chosen, rejected, estimator)
    assert_true(score["quality_score"] > 0.72, "simple separated valid pair should be high confidence")
    assert_true(not score["flags"]["chosen_syntax_error"], "chosen syntax flag incorrect")

    identical = p.score_pair(instruction, chosen, chosen, estimator)
    assert_true(identical["flags"]["chosen_equals_rejected"], "identical pair not flagged")
    assert_true(identical["flags"]["near_duplicate"], "identical pair not near-duplicate")


def test_split_overlap_is_reported_not_filtered() -> None:
    raw = []
    for idx in range(6):
        raw.append(
            {
                "_raw_index": idx,
                "_source_file": "synthetic.parquet",
                "_source_row": idx,
                "id": "same_prompt" if idx in {0, 5} else f"id_{idx}",
                "prompt": "same prompt" if idx in {0, 5} else f"prompt {idx}",
                "chosen": "```python\ndef f():\n    return 1\n```",
                "rejected": "```python\ndef f():\n    return 0\n```",
            }
        )
    shuffled = raw[:]
    random.Random(42).shuffle(shuffled)
    train = shuffled[2:]
    test = shuffled[:2]
    estimator = p.TokenLengthEstimator.create(None)
    train_views = p.build_train_views(
        train,
        prompt_template=p.DEFAULT_PROMPT_TEMPLATE,
        estimator=estimator,
        seed=42,
        quality_threshold=0.72,
        debug_size=128,
    )
    test_view = p.build_generation_test(test, prompt_template=p.DEFAULT_PROMPT_TEMPLATE)
    overlap = p.prompt_overlap_stats(p.metas_only(train_views["raw_items"]), p.metas_only(test_view["items"]))
    assert_true(overlap["prompt_overlap_allowed_by_project_policy"], "overlap policy marker missing")
    assert_true(overlap["overlap_unique_prompt_hashes"] >= 0, "overlap stat missing")


def test_dataset_info_contract() -> None:
    info = p.dataset_info()
    assert_true("code_dpo_a3_v2_high_confidence_train" in info, "missing high confidence registry")
    assert_true(info["code_dpo_a3_v2_high_confidence_train"]["ranking"] is True, "ranking flag missing")
    assert_true(info["code_dpo_a3_v2_generation_test"]["columns"]["response"] == "output", "generation response mapping wrong")


def main() -> int:
    test_extract_and_syntax()
    test_score_flags()
    test_split_overlap_is_reported_not_filtered()
    test_dataset_info_contract()
    print("A3 v2 unit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
