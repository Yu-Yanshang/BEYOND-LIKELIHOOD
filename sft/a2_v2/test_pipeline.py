from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sft.a2_v2 import pipeline


def dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data_root = root / "data_runs"
        self.output_root = root / "output_runs"
        self.model = root / "model"
        self.model.mkdir()
        train = [{"instruction": f"task {i}", "input": "", "output": "x" * (i + 1)} for i in range(400)]
        valid = [{"instruction": f"valid {i}", "input": "", "output": "def f(): pass"} for i in range(100)]
        test = [{"instruction": f"test {i}", "input": "", "output": "def f(): pass"} for i in range(100)]
        mbpp = [{"instruction": f"mbpp {i}", "input": "", "output": "def f(): pass", "task_id": i} for i in range(80)]
        self.sources = {}
        for name, rows in (("train", train), ("valid", valid), ("test", test), ("mbpp", mbpp)):
            path = root / f"{name}.json"
            dump(path, rows)
            self.sources[name] = path
        self.patches = [
            patch.object(pipeline, "DATA_RUN_ROOT", self.data_root),
            patch.object(pipeline, "OUTPUT_RUN_ROOT", self.output_root),
            patch.object(pipeline, "SOURCE_DATA", self.sources),
            patch.object(pipeline, "BASE_MODEL", self.model),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def test_acceptance_prepare_and_no_overwrite(self) -> None:
        pipeline.prepare_run("test", "acceptance", False)
        data = self.data_root / "test"
        self.assertEqual(len(pipeline.read_json(data / "a2_train.json")), 256)
        self.assertEqual(len(pipeline.read_json(data / "a2_valid.json")), 64)
        config = pipeline.read_json(data / "dataset_info.json")
        self.assertIn("a2_mbpp", config)
        qdora = pipeline.yaml.safe_load((self.output_root / "test/configs/train_qdora.yaml").read_text())
        self.assertEqual(qdora["max_steps"], 30)
        self.assertTrue(qdora["use_dora"])
        self.assertEqual(qdora["quantization_bit"], 4)
        with self.assertRaises(FileExistsError):
            pipeline.prepare_run("test", "acceptance", False)
        pipeline.prepare_run("test", "acceptance", True)

    def test_formal_has_three_epochs_without_max_steps(self) -> None:
        pipeline.prepare_run("formal", "formal", False)
        config = pipeline.yaml.safe_load((self.output_root / "formal/configs/train_full.yaml").read_text())
        self.assertEqual(config["num_train_epochs"], 3.0)
        self.assertNotIn("max_steps", config)
        self.assertEqual(config["template"], "qwen3")
        self.assertFalse(config["enable_thinking"])

    def test_code_metrics_helpers(self) -> None:
        self.assertEqual(pipeline.extract_code("text\n```python\ndef f():\n return 1\n```"), "def f():\n return 1")
        self.assertEqual(pipeline.first_function_name("def f():\n pass"), "f")
        self.assertGreater(pipeline.token_f1("def f(): return 1", "def f(): return 1"), 0.99)


if __name__ == "__main__":
    unittest.main()
