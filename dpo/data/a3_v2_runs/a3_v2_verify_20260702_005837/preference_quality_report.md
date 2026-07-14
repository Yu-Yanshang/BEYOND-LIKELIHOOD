# A3 v2 Preference Quality Report

- Score version: `a3_v2_quality_score_1.0`
- High-confidence threshold: `0.72`
- Token length mode: `qwen3_tokenizer`
- Prompt overlap allowed by project policy: `True`

## Data-view counts

| View | Rows |
|---|---:|
| `raw_total` | 9466 |
| `train_candidates` | 8966 |
| `test_candidates` | 500 |
| `raw_train` | 8924 |
| `clean_train` | 8924 |
| `high_confidence_train` | 8797 |
| `debug_train` | 128 |
| `generation_test` | 485 |

## Score distribution

```json
{
  "count": 8924,
  "min": 0.572654,
  "p10": 0.802016,
  "p25": 0.880751,
  "p50": 0.945531,
  "p75": 0.973205,
  "p90": 0.988637,
  "p95": 0.994052,
  "p99": 0.998734,
  "max": 1.0,
  "mean": 0.919177
}
```

## Pair-quality flags

```json
{
  "both_syntax_error": 89,
  "chosen_syntax_error": 202,
  "chosen_too_long": 33,
  "extreme_length_gap": 438,
  "low_separation": 1059,
  "near_duplicate": 297,
  "only_rejected_syntax_ok": 113,
  "rejected_syntax_error": 662,
  "rejected_too_long": 124,
  "response_too_short": 9
}
```

## Syntax proxy counts

```json
{
  "both_syntax_ok": 8149,
  "chosen_only_syntax_ok": 573,
  "neither_syntax_ok": 89,
  "rejected_only_syntax_ok": 113
}
```

## Prompt overlap

- Train unique prompt hashes: 5746
- Test unique prompt hashes: 477
- Overlap unique prompt hashes: 237

## Charts

- `reports/quality_score_hist.png`
- `reports/pair_similarity_hist.png`
- `reports/chosen_response_tokens_hist.png`
- `reports/rejected_response_tokens_hist.png`
- `reports/response_token_gap_hist.png`
- `reports/filter_funnel.png`

## Notes

- Quality score is a static proxy, not a proof of functional correctness.
- Prompt overlap is intentionally reported but not filtered for this training-run contract.
- Final code correctness should be measured by execution-based evaluation outside A3.
