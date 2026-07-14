# Test-Time Scaling (A5)

This module evaluates the selected A2 Full model without changing model weights.
It implements the four advanced requirements plus greedy and CoT baselines:

- `self_consistency`: independently sample multiple CoT paths, canonicalize the
  extracted Python AST, use majority agreement, then break ties with the verifier.
- `best_of_n`: sample N candidates and rank them by syntax, passed tests, complete
  task pass, and output format.
- `reflexion`: turn syntax/test failures into a correction prompt and retain the
  best verifier-scored version across rounds.
- `tot`: generate, heuristically score, expand, and prune intermediate approaches
  with beam search, then generate and verify code from the surviving paths.

Generated code is executed by the existing evaluator in an isolated temporary
directory with wall-clock, CPU, address-space, and file-size limits.

## Environment and default model

```bash
cd /root/project
conda activate shixun

MODEL=/root/project/best_path/runs/best_full_dpo_a5_20260702/models/a2_full
DATA=/root/mbpp/sanitized/test-00000-of-00001.parquet
```

## Quick smoke tests

Run one strategy on five tasks:

```bash
MODEL_PATH="$MODEL" STRATEGY=self_consistency NUM_SAMPLES=3 \
  OUTPUT_DIR=tts/outputs/tts_sc_debug \
  bash tts/run_test_time_scaling.sh --limit 5
```

```bash
MODEL_PATH="$MODEL" STRATEGY=best_of_n BEST_OF_N=5 \
  OUTPUT_DIR=tts/outputs/tts_bon_debug \
  bash tts/run_test_time_scaling.sh --limit 5
```

```bash
MODEL_PATH="$MODEL" STRATEGY=reflexion REFLEXION_ROUNDS=2 \
  OUTPUT_DIR=tts/outputs/tts_reflexion_debug \
  bash tts/run_test_time_scaling.sh --limit 5
```

```bash
MODEL_PATH="$MODEL" STRATEGY=tot TREE_WIDTH=3 TREE_DEPTH=2 \
  OUTPUT_DIR=tts/outputs/tts_tot_debug \
  bash tts/run_test_time_scaling.sh --limit 5
```

## Full comparison

The comparison runs greedy, CoT, Self-Consistency, Best-of-N, Reflexion, and
Tree of Thoughts with the same model and test set. It can take several hours.

```bash
MODEL_PATH="$MODEL" STRATEGY=compare \
NUM_SAMPLES=5 BEST_OF_N=5 REFLEXION_ROUNDS=2 TREE_WIDTH=3 TREE_DEPTH=2 \
OUTPUT_DIR=tts/outputs/tts_a2_full_compare \
bash tts/run_test_time_scaling.sh
```

For a detached run:

```bash
nohup env MODEL_PATH="$MODEL" STRATEGY=compare \
  NUM_SAMPLES=5 BEST_OF_N=5 REFLEXION_ROUNDS=2 TREE_WIDTH=3 TREE_DEPTH=2 \
  OUTPUT_DIR=tts/outputs/tts_a2_full_compare \
  bash tts/run_test_time_scaling.sh \
  > tts/outputs/tts_a2_full_compare.log 2>&1 &
```

Monitor it with:

```bash
tail -f tts/outputs/tts_a2_full_compare.log
```

## Outputs

Each strategy writes `<output_dir>/<strategy>/metrics.json` and `cases.jsonl`.
The output root also contains `comparison.json`, `comparison.csv`, and
`comparison.md`. Metrics include pass@1, syntax/test pass rates, elapsed time,
generated sequence/token counts, and peak allocated GPU memory. Strategy-specific
fields record majority consistency, correction rate, or tree search cost.
