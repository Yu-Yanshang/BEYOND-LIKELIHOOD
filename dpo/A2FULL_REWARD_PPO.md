# A2-full reward-function PPO

This run path starts from:

```text
/root/project/best_path/runs/best_full_dpo_a5_20260702/models/a2_full
```

It trains only a small LoRA PPO adapter under `dpo/outputs/<RUN_ID>/ppo_adapter`.
The reference model for PPO KL is the same A2-full model, so the policy is
penalized for drifting away from the strongest available checkpoint.

## Reward design

The reward API is implemented in:

```text
dpo/scripts/a2full_ppo_reward.py
dpo/scripts/a2full_ppo_reward_server.py
```

For MBPP-style prompts containing `assert` tests, the reward executes the
generated candidate in a temporary subprocess with timeout and memory limits.
The largest positive reward comes from passing prompt tests. The fallback
reward covers Python syntax, safe imports/calls, callable structure, reasonable
length, and task-keyword relevance. Refusals, placeholders, unsafe operations,
top-level interactive I/O, echoed tests, syntax errors, and overlong/empty code
are penalized. Scores are clipped to `[-3, 3]`.

The stricter reward also penalizes leaked `<think>` / `</think>` tags, chat
special tokens, explanatory preambles, copied asserts, repeated lines/ngrams,
sample `print(...)` calls, missing test-called function names, and responses
that become too long and likely to truncate.

## Loss and anti-forgetting design

The training config uses LLaMA-Factory/TRL PPO:

- clipped PPO policy loss
- value loss from the value head
- adaptive KL penalty against the A2-full reference model
- reward normalization and whitening

Additional anti-forgetting controls:

- LoRA-only updates on `q_proj,k_proj,v_proj,o_proj`
- low rank `r=8`, dropout `0.05`, learning rate `5e-7`
- `ppo_target: 1.5` to keep KL tight
- MBPP baseline guard after training; by default at least one guarded metric
  must improve, not merely avoid a large regression

The guard compares PPO against the starting A2-full model on MBPP and fails the
run if pass rate, average test pass rate, or syntax rate drops beyond the
configured tolerances. Set `REQUIRE_ANY_GAIN=0` only for diagnostic runs where
"no obvious regression" is enough.

## Run

Foreground:

```bash
cd /root/project
RUN_ID=a2full_best_ppo_reward_v2_20260709 GPU_ID=0 \
  bash dpo/scripts/run_a2full_best_ppo_reward.sh
```

Background:

```bash
cd /root/project
RUN_ID=a2full_best_ppo_reward_v2_20260709 GPU_ID=0 \
  bash dpo/scripts/submit_a2full_best_ppo_reward.sh
```

Fast smoke run:

```bash
cd /root/project
RUN_ID=a2full_best_ppo_reward_v2_smoke MAX_TRAIN_SAMPLES=32 MAX_STEPS=2 EVAL_LIMIT=10 \
  bash dpo/scripts/run_a2full_best_ppo_reward.sh
```

Useful outputs:

```text
dpo/outputs/<RUN_ID>/TRAINING_RECORD.md
dpo/outputs/<RUN_ID>/rendered_configs/ppo_reward.yaml
dpo/outputs/<RUN_ID>/logs/reward_scores.jsonl
dpo/outputs/<RUN_ID>/ppo_adapter/
dpo/outputs/<RUN_ID>/mbpp_eval/
dpo/outputs/<RUN_ID>/ppo_guard_report.json
```
