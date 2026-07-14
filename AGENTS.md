# AGENTS.md

This file is the operational guide for agents working in `/root/project`.

## Project and runtime

The project implements a Qwen3-0.6B Python code-generation training chain. The
active environment and paths are:

- Project root: `/root/project`
- Conda environment: `/opt/conda/envs/shixun`
- Base model: `/root/project/Qwen3-0.6B`
- LLaMA-Factory: `/root/project/LlamaFactory`
- Proposal: `/root/project/docs/a2_a3_proposal.md`
- Shared runtime helper: `/root/project/scripts/common_env.sh`
- A2 v2 runtime helper: `/root/project/sft/a2_v2/common_env.sh`

The GPU is one H200 NVL partition with about 23 GiB visible memory. PyTorch is
`2.12.1+cu130`. `bitsandbytes==0.49.2` is installed in `shixun`; A2 scripts add
the CUDA 13 library directory and `BNB_CUDA_VERSION=130` automatically.

Start ordinary project work with:

```bash
cd /root/project
conda activate shixun
```

## Active module state

- A1 data preparation exists under `sft/data/`; treat train/valid/test JSON as
  read-only module inputs.
- A2 v2 is implemented under `sft/a2_v2/`, with isolated Full SFT, QLoRA, and
  QDoRA configurations and versioned runs.
- A3 v2 is implemented under `dpo/a3_v2/`; its default A4 input is the
  `code_dpo_a3_v2_high_confidence_train` view.
- Existing default SFT/DPO scripts remain historical baselines. Do not replace
  or silently redirect them to v2 implementations.
- A4 remains a downstream module, but both A3 and A2 provide one-step bridge
  validators proving trainer compatibility.

Qwen3 training and evaluation use:

```yaml
template: qwen3
enable_thinking: false
bf16: true
fp16: false
```

Do not enable thinking for the current A1/A3 datasets; they contain no thinking
targets. A separate inference ablation may enable it explicitly.

## A2 v2 commands

Acceptance runs Full, QLoRA, and QDoRA for 30 optimizer steps, evaluates fixed
32-sample A1/MBPP subsets, exports PEFT adapters, and runs all three A4 bridges:

```bash
RUN_ID=a2_accept_001 A3_RUN_ID=<a3_run_id> \
  bash sft/scripts/run_a2_v2.sh acceptance
```

The complete experiment runs all variants for 3 epochs and performs checkpoint
selection and full evaluation:

```bash
RUN_ID=a2_formal_001 A3_RUN_ID=<a3_run_id> \
  bash sft/scripts/run_a2_v2.sh formal
```

Detached submission and log access:

```bash
RUN_ID=a2_formal_001 A3_RUN_ID=<a3_run_id> \
  bash sft/scripts/submit_a2_v2.sh formal
tail -f sft/outputs/a2_v2_runs/a2_formal_001/logs/a2_formal.log
```

Separate variant entry points are available after `prepare`:

```bash
RUN_ID=<id> PROFILE=formal bash sft/scripts/run_a2_v2.sh prepare
RUN_ID=<id> bash sft/scripts/train_a2_full.sh
RUN_ID=<id> bash sft/scripts/train_a2_qlora.sh
RUN_ID=<id> bash sft/scripts/train_a2_qdora.sh
```

Existing run directories are protected. Use `RESUME=1` only to continue the
same run and profile. Do not delete stage markers merely to force overwrites.

## Interfaces and outputs

A2 data and outputs are isolated by run ID:

```text
sft/data/a2_v2_runs/<RUN_ID>/
sft/outputs/a2_v2_runs/<RUN_ID>/
```

Important A2 outputs:

- `a2_handoff_manifest.json`: resolved Full/adapter/merged paths for A4/A5.
- `selected_models.json`: checkpoint-selection decision.
- `reports/comparison_report.{json,csv,md}` and comparison plot.
- `reports/artifact_change_validation.json`: proof that all trainable weights changed.
- `a4_bridge/bridge_summary.json`: A2→A3→A4 integration result.
- `logs/`, `telemetry/`, and rendered `configs/` for reproducibility.

A3 data and logs use:

```text
dpo/data/a3_v2_runs/<RUN_ID>/
dpo/outputs/a3_v2_runs/<RUN_ID>/
```

The intended downstream data flow is:

```text
A1 JSON → A2 selected/merged model
A2 model + A3 high-confidence preference data → A4 preference training
```

## Compatibility notes

Use `python -m sft.a2_v2.llamafactory_compat` through the A2 scripts. The local
launcher fixes two runtime-only enum/API mismatches without editing vendored
LLaMA-Factory or TRL:

1. Transformers 5 names BNB quantization `bitsandbytes`, while this
   LLaMA-Factory revision compares against `bnb`, otherwise rejecting QDoRA.
2. TRL 0.24 interprets Transformers 5 optional-package availability tuples as
   truthy, otherwise trying to import absent `mergekit` during DPO startup.

Do not apply these fixes directly to `LlamaFactory/` or site-packages unless the
framework versions are deliberately upgraded and all A2/A3 bridge tests rerun.

## Verified evidence and boundaries

The non-final acceptance run `a2_v2_accept_20260702_impl` completed 30 steps for
Full, QLoRA, and QDoRA, adapter merges, fixed-subset prediction/evaluation, and
three A4 one-step probes. It proves runtime viability only; it is not a
substitute for the formal 3-epoch comparison.

Preserve these boundaries:

- Do not overwrite A1 data, default SFT/DPO configs, historical outputs, or A3
  v2 runs.
- Do not treat inherited or acceptance logs as final project metrics.
- Do not download another base model unless explicitly requested.
- Keep generated configs, models, logs, and reports inside their module's
  versioned run directory.
- Run generated-code execution evaluation with time and memory limits.
