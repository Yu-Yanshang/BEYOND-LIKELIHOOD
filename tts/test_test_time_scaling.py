#!/usr/bin/env python3
"""CPU-only unit tests for Test-Time Scaling selection helpers."""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_time_scaling import canonical_code, score_thought, select_best


def candidate(candidate_id, score, passed_tests=0, syntax=True, code="pass"):
    return {
        "candidate_id": candidate_id,
        "verifier_score": score,
        "passed_tests": passed_tests,
        "syntax_ok": syntax,
        "final_code": code,
    }


class TestSelection(unittest.TestCase):
    def test_canonical_code_normalizes_formatting(self):
        left = canonical_code("def f(x):\n return x+1")
        right = canonical_code("def f(x):\n    return x + 1\n")
        self.assertEqual(left, right)
        ast.parse(left)

    def test_select_best_prefers_verifier_score(self):
        selected = select_best([candidate(0, 2.0), candidate(1, 8.0, passed_tests=2)])
        self.assertEqual(selected["candidate_id"], 1)

    def test_thought_penalizes_code_in_planning_stage(self):
        task = "Write a function add_one(x)."
        plan = "Use a direct arithmetic return. Check integer and float edge cases. Complexity O(1)."
        code = "```python\ndef add_one(x): return x + 1\n```"
        self.assertGreater(score_thought(plan, task, 1), score_thought(code, task, 1))


if __name__ == "__main__":
    unittest.main()
