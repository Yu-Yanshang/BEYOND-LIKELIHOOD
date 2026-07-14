#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from evaluation_catalog import discover_metric_records, infer_training_method, merged_training_runs


class TestEvaluationCatalog(unittest.TestCase):
    def test_discovers_and_normalizes_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "tts" / "outputs" / "run" / "best_of_n" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "strategy": "best_of_n",
                        "total": 257,
                        "pass_at_1": 0.6,
                        "syntax_pass_rate": 1.0,
                        "avg_test_pass_rate": 0.7,
                    }
                ),
                encoding="utf-8",
            )
            records = discover_metric_records(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].module, "tts")
            self.assertEqual(records[0].strategy, "best_of_n")
            self.assertFalse(records[0].is_smoke)

    def test_marks_small_runs_as_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "best_path" / "eval" / "probe" / "metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"total": 2, "pass_at_1": 1.0}), encoding="utf-8")
            record = discover_metric_records(root)[0]
            self.assertTrue(record.is_smoke)

    def test_merges_train_and_eval_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "sft" / "outputs" / "model"
            model.mkdir(parents=True)
            (model / "train_results.json").write_text(
                json.dumps({"epoch": 2, "train_loss": 0.5, "train_runtime": 100}), encoding="utf-8"
            )
            (model / "eval_results.json").write_text(json.dumps({"eval_loss": 0.6}), encoding="utf-8")
            merged = merged_training_runs(discover_metric_records(root))
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["train_loss"], 0.5)
            self.assertEqual(merged[0]["eval_loss"], 0.6)

    def test_leaf_variant_wins_over_run_name(self):
        self.assertEqual(
            infer_training_method(
                "best_path/runs/best_full_dpo_a5/eval/a2_full/metrics.json", {}
            ),
            "Full SFT",
        )
        self.assertEqual(
            infer_training_method(
                "tts/outputs/tts_self_consistency/self_consistency/metrics.json",
                {"model_path": "/root/project/best_path/runs/best_full_dpo_a5/models/a2_full"},
            ),
            "Full SFT",
        )
        self.assertEqual(
            infer_training_method(
                "dpo/outputs/a2full_dpo_lora_ppo/mbpp_eval/a2_full/mbpp_metrics.json", {}
            ),
            "Full SFT",
        )


if __name__ == "__main__":
    unittest.main()
