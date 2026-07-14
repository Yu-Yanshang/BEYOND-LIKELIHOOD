#!/usr/bin/env python3
"""CPU-only tests for the showcase CLI helpers."""

import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from showcase_cli import (
    EXAMPLES,
    best_path_messages,
    best_path_repair_messages,
    build_row,
    consensus_counts,
    hybrid_score,
    performance_rows,
    recent_sessions,
)


def candidate(code, score):
    return {"final_code": code, "verifier_score": score}


class TestShowcaseHelpers(unittest.TestCase):
    def test_examples_have_executable_asserts(self):
        self.assertGreaterEqual(len(EXAMPLES), 4)
        for example in EXAMPLES.values():
            self.assertTrue(example["task"])
            self.assertTrue(all(test.startswith("assert ") for test in example["tests"]))

    def test_build_row_attaches_tests_to_prompt_and_evaluator(self):
        row = build_row("Write f(x).", ["assert f(1) == 1"])
        self.assertIn("assert f(1) == 1", row["task"])
        self.assertEqual(row["tests"], ["assert f(1) == 1"])

    def test_best_path_prompt_contains_task_and_visible_tests(self):
        row = build_row("Write f(x).", ["assert f(1) == 1"])
        messages = best_path_messages(row)
        self.assertIn("precise Python", messages[0]["content"])
        self.assertIn("Visible tests", messages[1]["content"])
        self.assertIn("assert f(1) == 1", messages[1]["content"])

    def test_repair_prompt_identifies_exact_failed_test(self):
        row = build_row("Write f(x).", ["assert f(1) == 1"])
        previous = {
            "final_code": "def f(x): return 0",
            "test_results": [{"passed": False, "stderr": "AssertionError"}],
        }
        messages = best_path_repair_messages(row, previous, 1)
        self.assertIn("Test: assert f(1) == 1", messages[1]["content"])

    def test_consensus_bonus_rewards_agreement(self):
        candidates = [
            candidate("def f(x):\n return x", 3.0),
            candidate("def f(x):\n    return x", 3.0),
            candidate("def f(x):\n return x + 1", 3.0),
        ]
        counts = consensus_counts(candidates)
        self.assertGreater(
            hybrid_score(candidates[0], counts, len(candidates)),
            hybrid_score(candidates[2], counts, len(candidates)),
        )

    def test_performance_uses_best_path_a2_routes(self):
        rows = performance_rows()
        if rows:
            self.assertEqual([row["label"] for row in rows], ["A2 Full · greedy", "A2 Full · A5 enhanced"])

    def test_recent_sessions_returns_valid_newest_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "broken.json").write_text("{", encoding="utf-8")
            payload = {
                "created_at": "2026-07-12T10:00:00+08:00",
                "strategy": "greedy",
                "task": "Write f(x).",
                "metrics": {"elapsed_seconds": 1.25},
                "result": {
                    "syntax_ok": True,
                    "passed": True,
                    "passed_tests": 1,
                    "total_tests": 1,
                },
            }
            (root / "session.json").write_text(json.dumps(payload), encoding="utf-8")
            sessions = recent_sessions(root, limit=3)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["strategy"], "greedy")
            self.assertEqual(sessions[0]["passed_tests"], 1)


if __name__ == "__main__":
    unittest.main()
