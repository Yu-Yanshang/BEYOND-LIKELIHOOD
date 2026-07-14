# Fine-Tuning and Preference Data Construction for a Qwen3-Based Python Code Generation System 

## Personal Proposal for Modules A2 and A3

![System Architecture Placeholder](https://cdn.luogu.com.cn/upload/image_hosting/9v8bax3d.png)

## 1. Basic Information

| Item             | Content                                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Student Name     | Su Kun                                                                                                                           |
| Student ID       | 20235805                                                                                                                         |
| Project Name     | Large-Model Fine-Tuning and Inference Enhancement System for Python Code Generation                                              |
| Selected Modules | A2: Supervised Fine-Tuning; A3: Preference Data Construction                                                                     |
| Personal Role    | Owner of the first-stage model adaptation pipeline and the preference-data foundation for downstream alignment                   |

## 2. Project-Level Objective and Personal Positioning

This project aims to build a complete optimization pipeline for Python code generation. Given a natural-language programming task, the system should generate executable Python code that satisfies the requested function signature, handles boundary cases, avoids irrelevant explanations, and passes execution-based tests. The group system is organized as a modular pipeline: instruction data preparation, supervised fine-tuning, preference data construction, preference alignment, inference-time enhancement, and unified evaluation.

My responsibility covers two structurally critical modules:

1. **A2: Supervised Fine-Tuning.** This module transforms a general base model into a model specialized for the project’s Python coding instruction distribution.
2. **A3: Preference Data Construction.** This module converts raw chosen/rejected preference pairs into reliable alignment data for DPO-style training and its extensions.

My personal positioning is not only to complete two functional modules, but to make them the technical backbone of the group’s final system. A2 determines the quality of the policy initialization used by later alignment, while A3 determines whether downstream preference optimization learns a meaningful quality signal or merely amplifies noise. Therefore, the proposal emphasizes data contracts, ablation design, resource-aware training, Qwen3 migration, QDoRA experimentation, and execution-aware evaluation.

## 3. Overall System Design

The proposed system uses a **Qwen3-0.6B-centered training and evaluation chain** rather than treating the default Qwen1.5-0.5B-Chat model as the only baseline. The default model remains useful as a reference point, but the main deliverable will be organized around Qwen3 because it is newer, small enough for the training environment, and better aligned with modern reasoning-oriented LLM workflows.

### 3.1 High-Level Flow

| Stage | Owner    | Input                       | Output                                                                 | Role in the System                           |
| ----- | -------- | --------------------------- | ---------------------------------------------------------------------- | -------------------------------------------- |
| A1    | Teammate | Raw instruction data        | SFT train/valid/test JSON                                              | Provides supervised samples                  |
| A2    | Me       | SFT data + Qwen3 base model | SFT checkpoint/adapters, predictions, metrics                          | Produces the first adapted policy model      |
| A3    | Me       | Raw py-dpo preference data  | Ranking-format data, generation-format test data, data-quality reports | Provides alignment-ready preference data     |
| A4    | Teammate | A2 model + A3 data          | DPO/SimPO/ORPO models                                                  | Improves preference alignment                |
| A5    | Teammate | Base/SFT/DPO models         | CoT, Best-of-N, Reflexion, verifier reranking results                  | Improves inference without parameter updates |
| A6    | Group    | Outputs from all modules    | Unified report, demo, metric tables, error analysis                    | Presents integrated project quality          |

### 3.2 Architectural Principles

The design follows four principles.

**Modularity.** A2 and A3 will each have independent inputs, outputs, configurations, logs, and demonstration paths. They can be evaluated separately without relying on a Web Demo.

**Contract-first integration.** Outputs will follow stable file contracts: model directories for A2, JSON/JSONL data contracts for A3, and metrics/cases files for A6.

**Experiment-driven credibility.** The proposal avoids a single-run demonstration. It defines baselines, ablations, resource measurements, and failure analysis.

**Modern but feasible innovation.** Qwen3 migration and QDoRA will be treated as controlled engineering upgrades, not as uncontrolled novelty. Every innovation must have a fallback path.

## 4. Module A2: Supervised Fine-Tuning

![A2 Training Strategy Placeholder](https://cdn.luogu.com.cn/upload/image_hosting/2zfu5lxj.png)

### 4.1 Module Necessity

A general chat model may understand Python syntax, but it is not automatically optimized for the project’s exact code-generation distribution. Typical failure modes include misunderstanding the required function signature, mixing explanations with executable code, omitting boundary cases, generating incomplete logic, or producing output that is difficult for an evaluator to extract. SFT is necessary because it directly teaches the model the expected mapping from task instruction to Python solution format.

A2 is also a prerequisite for stronger downstream alignment. A poor SFT policy gives A4 a weak starting point; DPO-style algorithms can refine preferences but cannot reliably compensate for a model that has not first learned the basic task format.

### 4.2 Module Tasks

A2 will provide the following capabilities:

| Task                            | Purpose                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3 base-model validation     | Confirm that the chosen base can load, infer, and perform a small training smoke test in the offline environment                |
| Full SFT baseline               | Establish a strong reference model for supervised adaptation                                                                    |
| QLoRA baseline                  | Measure the trade-off among memory cost, training speed, and code-generation quality                                            |
| QDoRA extension                 | Test whether DoRA under a QLoRA-style interface improves low-rank adaptation quality                                            |
| Batch prediction and evaluation | Generate test-set outputs and compare them with the base model and later aligned models                                         |
| Training artifact governance    | Save checkpoints, adapter states, tokenizer state, generation outputs, metrics, and logs under unambiguous Qwen3-prefixed names |

### 4.3 Technical Challenges

**Model migration risk.** Qwen3 requires correct tokenizer/template handling and correct precision settings. A mismatched chat template can silently degrade training and evaluation.

**Resource constraints.** Full fine-tuning is attractive for quality but has higher memory and time cost. QLoRA and QDoRA are planned as resource-aware alternatives.

**Output-format discipline.** Code generation evaluation requires clean Python extraction. The model must be discouraged from producing excessive explanation when the evaluator expects executable code.

**Fair comparison.** Base, SFT, QLoRA, QDoRA, and later DPO models must be compared under the same dataset split, generation parameters, and MBPP-style execution protocol.

**Reproducibility.** Random seeds, data versions, model paths, precision mode, and generation settings must be logged to avoid irreproducible leaderboard claims.

### 4.4 Proposed A2 Innovation Points

#### 4.4.1 Qwen3-Centered Base Model Upgrade

The main training chain will use Qwen3-0.6B as the upgraded base model. This is not a cosmetic replacement: it changes the expected tokenizer, template, precision preference, output behavior, and experiment naming convention. The project will retain Qwen1.5 as a historical baseline only where useful.

#### 4.4.2 Three-Level Training Ladder

A2 will be designed as a progressive ladder:

| Level | Method                | Purpose                                                                 |
| ----- | --------------------- | ----------------------------------------------------------------------- |
| L0    | Qwen3 Base Evaluation | Establish an untrained Qwen3 baseline                                   |
| L1    | Full SFT              | Maximize supervised adaptation capacity                                 |
| L2    | QLoRA SFT             | Reduce memory cost through 4-bit quantized training with LoRA adapters  |
| L3    | QDoRA SFT             | Add DoRA’s magnitude-direction decomposition to the QLoRA-style setting |

This ladder allows the final report to explain not merely that SFT was performed, but how different adaptation regimes affect accuracy, stability, and resource usage.

#### 4.4.3 QDoRA as a Controlled Extension

QDoRA will be framed as a parameter-efficient innovation rather than a replacement for all baselines. The expected comparison is:

* QLoRA: lower memory, ordinary low-rank update.
* QDoRA: similar interface, additional magnitude learning, potentially stronger adaptation at low rank.
* Full SFT: upper-reference training route if resources allow.

The decision criterion will not be “newer is better,” but whether QDoRA improves execution-based quality or stability under the same budget.

#### 4.4.4 Execution-Aware Model Selection

The best checkpoint should not be selected only by validation loss. For code generation, the final model selection should also consider syntax correctness, test pass rate, average generated length, and failure categories. This avoids selecting a low-loss model that produces verbose or non-executable responses.

### 4.5 A2 Input and Output Contract

| Type               | Planned Content                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------- |
| Input data         | `sft/data/code_sft_train.json`, `sft/data/code_sft_valid.json`, `sft/data/code_sft_test.json` |
| Base model         | Project-local Qwen3-0.6B directory                                                            |
| Configuration      | YAML-level training configurations for full SFT, QLoRA, and QDoRA                             |
| Model output       | Qwen3 SFT checkpoint or adapter directory                                                     |
| Prediction output  | Generated test-set predictions in JSONL format                                                |
| Metric output      | MBPP-style execution metrics, syntax metrics, latency, generated length, and error cases      |
| Integration output | Model path and metrics file consumable by A4, A5, and A6                                      |

### 4.6 A2 Independent Demonstration Plan

The A2 demonstration will show:

1. A clean environment and base-model loading check.
2. A mini training or checkpoint-loading proof.
3. A single-prompt inference example.
4. Batch prediction on a fixed test subset.
5. Metric comparison between Qwen3 Base, Qwen3 Full SFT, Qwen3 QLoRA, and Qwen3 QDoRA if time permits.
6. Error cases categorized by syntax error, signature mismatch, incomplete logic, boundary-case failure, or verbose non-code output.

## 5. Module A3: Preference Data Construction

![A3 Data Pipeline Placeholder](https://cdn.luogu.com.cn/upload/image_hosting/0oda36q0.png)

### 5.1 Module Necessity

Preference optimization is only as reliable as the chosen/rejected pairs it consumes. A raw preference dataset can contain missing fields, duplicated prompts, extremely short or malformed answers, length-biased pairs, near-identical pairs, or pairs where the chosen response is not truly better for executable code. If these issues are not handled, DPO-style training may learn formatting artifacts instead of real code quality.

A3 is therefore not a simple format-conversion module. It is the quality gate between raw preference data and model alignment.

### 5.2 Module Tasks

| Task                     | Purpose                                                              |
| ------------------------ | -------------------------------------------------------------------- |
| Raw preference ingestion | Read the local py-dpo Parquet source                                 |
| Field normalization      | Map prompt/chosen/rejected into LLaMA-Factory ranking format         |
| Data cleaning            | Remove missing, malformed, duplicate, or unusable samples            |
| Pair-quality auditing    | Estimate whether each pair contains a meaningful preference signal   |
| Split construction       | Create leakage-resistant train/test splits                           |
| Multi-format output      | Produce ranking-format training data and generation-format test data |
| Reporting                | Provide statistics, examples, bad cases, and data cards for A4/A6    |

### 5.3 Technical Challenges

**Preference noise.** Chosen responses are not guaranteed to be executable or strictly better than rejected responses for every task.

**Length bias.** A model may learn that longer answers are better even when shorter code is cleaner and more correct.

**Prompt leakage.** Similar prompts across train and test splits can inflate downstream evaluation.

**Format mismatch.** DPO training requires ranking samples, while evaluation often requires generation-format prompts and reference outputs.

**Downstream compatibility.** A3 must serve DPO, and should also leave room for SimPO, ORPO, KTO, or verifier-based extensions.

### 5.4 Proposed A3 Innovation Points

#### 5.4.1 Preference Data Quality Scorecard

A3 will produce a quality report beyond raw sample counts. Planned dimensions include:

| Dimension                         | Rationale                                                    |
| --------------------------------- | ------------------------------------------------------------ |
| Valid field ratio                 | Measures basic dataset integrity                             |
| Duplicate and near-duplicate rate | Prevents data leakage and over-counting                      |
| Prompt length distribution        | Detects abnormal task descriptions                           |
| Chosen/rejected length gap        | Exposes length bias                                          |
| Code block extractability         | Measures whether answers contain usable Python code          |
| Syntax-check proxy                | Flags responses likely to fail execution immediately         |
| Pair separability                 | Detects cases where chosen and rejected are nearly identical |
| Split leakage risk                | Checks whether similar prompts appear across splits          |

This scorecard gives A3 an independent demonstration value even before A4 training is completed.

#### 5.4.2 Alignment-Ready Data Contract

The output will not be only `code_dpo_train.json`. A3 will also provide sidecar metadata:

* dataset version,
* split seed,
* sample counts before and after cleaning,
* filtering reasons,
* bad-pair examples,
* representative high-confidence examples,
* quality flags that A4 may use for ablation.

This makes the module auditable and easier to defend during inspection.

#### 5.4.3 Quality-Stratified Ablation Sets

To show deeper engineering thinking, A3 will prepare multiple data views:

| Data View               | Purpose                                                 |
| ----------------------- | ------------------------------------------------------- |
| Raw converted set       | Baseline conversion result                              |
| Clean set               | Removes missing, duplicated, and malformed entries      |
| High-confidence set     | Keeps pairs with stronger format and quality indicators |
| Debug subset            | Supports fast training smoke tests                      |
| Held-out generation set | Supports fair model-level evaluation                    |

This allows the group to test whether better data curation improves DPO stability and downstream code quality.

#### 5.4.4 Future-Proof Preference Schema

Although A4 may initially use DPO, A3 will be designed to support later preference algorithms. For DPO/SimPO/ORPO-style methods, chosen/rejected pairs are sufficient. For KTO-style extensions, data may need to be transformed into desirable/undesirable examples. For reward-style experiments, unit-test or syntax-based metadata can become auxiliary supervision.

### 5.5 A3 Input and Output Contract

| Type               | Planned Content                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Input              | `py-dpo-v0.1/py-dpo.parquet`                                                                      |
| Required fields    | `id`, `prompt`, `chosen`, `rejected`                                                              |
| Training output    | `dpo/data/code_dpo_train.json` in ranking format                                                  |
| Test output        | `dpo/data/code_dpo_test.json` in generation format                                                |
| Registry output    | `dpo/data/dataset_info.json`                                                                      |
| Audit output       | `sample_preview.json`, `bad_pairs.json`, `data_statistics.json`, `preference_quality_report.json` |
| Integration output | Dataset name, split information, statistics, and quality flags for A4/A6                          |

### 5.6 A3 Independent Demonstration Plan

The A3 demonstration will show:

1. Number of raw samples and valid samples after each filtering stage.
2. Example conversion from raw prompt/chosen/rejected to ranking format.
3. Distribution charts for prompt length, chosen length, rejected length, and length gaps.
4. Examples of rejected bad pairs with explicit filtering reasons.
5. Final dataset registration and downstream compatibility check.
6. A small debug subset used to prove that the downstream trainer can read the constructed data.

## 6. Technical Stack

| Category              | Selection                                                                      | Rationale                                                                            |
| --------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Programming Language  | Python 3, shell scripts                                                        | Standard for data processing, training orchestration, and reproducible CLI workflows |
| Training Framework    | Local LLaMA-Factory 0.9.6.dev0                                                 | Provides the unified workflow for SFT, DPO-style training, LoRA, QLoRA, and DoRA-related extensions |
| Deep Learning Backend | PyTorch, Transformers                                                          | Model loading, training, generation, and tokenizer handling                          |
| PEFT Stack            | PEFT, bitsandbytes                                                             | Supports LoRA/QLoRA and DoRA-style adapter experiments                               |
| Base Model            | Qwen3-0.6B                                                                     | Small enough for the project environment while representing a modern Qwen generation |
| Algorithms            | Full SFT, QLoRA, QDoRA; DPO-ready preference construction                      | Balances quality, memory constraints, and alignment extensibility                    |
| Data Processing       | pandas, pyarrow, JSON/JSONL                                                    | Efficient Parquet ingestion and training-data export                                 |
| Evaluation            | MBPP-style execution benchmark, syntax checks, token-level similarity, latency | Code generation must be judged by executability, not only text similarity            |
| Storage               | Local model directories, JSON/JSONL, metrics files, image plots                | Simple, inspectable, and compatible with all group modules                           |
| Monitoring            | Training logs, loss curves, memory logs, quality reports                       | Required for reproducibility and convincing final analysis                           |

## 7 Runtime Environment

### 7.1 GPU Visibility and Memory Assumption

Although the physical accelerator is reported as **NVIDIA H200 NVL**, PyTorch currently exposes only one visible GPU with approximately **23 GiB** of visible memory. Therefore, all training and evaluation scripts should assume a single-GPU setting.

The 23 GiB root-view memory budget leads to the following planning assumptions:

| Task                   | Default Configuration                                               | Feasibility Under 23 GiB Root View                                            |
| ---------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Qwen3-0.6B Full SFT    | bf16, batch size 1–2, gradient accumulation 4–8, cutoff length 2048 | Feasible in principle, but peak memory must be measured                       |
| Qwen3-0.6B Full DPO    | policy + reference model, batch size 1, gradient accumulation 8     | Higher memory risk than SFT; not the first-choice path                        |
| LoRA / QLoRA / QDoRA   | PEFT with optional 4-bit quantization                               | Preferred for stable experiments under constrained memory                     |

### 7.2 Project Directory and Conda Environment

The unified project-root convention is:

```bash
PROJECT_ROOT=/root/project
LLAMA_FACTORY_DIR=$PROJECT_ROOT/LlamaFactory
```

The recommended Python interpreter is:

```bash
PYTHON_BIN=/opt/conda/envs/shixun/bin/python
```

The core project structure is expected to be:

```text
project/
  Qwen3-0.6B/
  LlamaFactory/
  python_code_instructions_18k_alpaca/
  py-dpo-v0.1/
  sft/
    configs/
    data/
    scripts/
    outputs/
  dpo/
    configs/
    data/
    scripts/
    outputs/
  tts/
    cot_code_example.py
    evaluate_cot_code.py
    run_cot_code_eval.sh
```

Historical output directories should not be used as evidence for final results. All newly generated outputs should use Qwen3-prefixed names to avoid contamination from Qwen1.5 experiments.

## 8. Experimental Plan

![Experiment Matrix Placeholder](https://cdn.luogu.com.cn/upload/image_hosting/rvwn9f4c.png)

### 8.1 Environment Preflight

| Check                  | Success Criterion                                                 |
| ---------------------- | ----------------------------------------------------------------- |
| Local model path check | Qwen3 model files and tokenizer files are readable                |
| Tokenizer check        | Chat template can be applied without error                        |
| Minimal inference      | A short prompt produces a short completion                        |
| Mini training check    | One to three training steps finish without OOM                    |
| Precision check        | bf16 is preferred when stable; fallback precision is documented   |
| Path check             | No script depends on obsolete `/data` paths or Qwen1.5-only names |

### 8.2 A2 Experiments

| Experiment               | Compared Models                        | Main Metrics                                                           |
| ------------------------ | -------------------------------------- | ---------------------------------------------------------------------- |
| Base comparison          | Qwen1.5 default baseline vs Qwen3 base | pass_at_1, syntax_pass_rate, avg_test_pass_rate                        |
| Supervised adaptation    | Qwen3 Base vs Qwen3 Full SFT           | execution metrics, validation loss, error distribution                 |
| PEFT efficiency          | Full SFT vs QLoRA                      | GPU memory, training speed, trainable parameters, final metrics        |
| QDoRA extension          | QLoRA vs QDoRA                         | same budget quality, stability, inference readiness                    |
| Output-format robustness | SFT variants under fixed prompts       | code extraction rate, verbose-output rate, function-signature accuracy |

### 8.3 A3 Experiments

| Experiment            | Compared Data Views                   | Main Metrics                                               |
| --------------------- | ------------------------------------- | ---------------------------------------------------------- |
| Cleaning impact       | Raw converted vs clean set            | valid ratio, duplicate rate, bad-pair rate                 |
| Quality filtering     | Clean set vs high-confidence set      | pair separability, code extractability, syntax proxy       |
| Split safety          | Random split vs leakage-checked split | prompt overlap, near-duplicate risk                        |
| Downstream smoke test | Debug subset consumed by A4 trainer   | trainer readability and early loss sanity                  |
| Data-card analysis    | All data views                        | length distributions, category examples, filtering reasons |

### 8.4 Test Scenarios

The fixed evaluation suite should include:

* simple function synthesis,
* list/string/dictionary processing,
* numerical edge cases,
* empty-input and boundary-input handling,
* required function-signature compliance,
* tasks requiring loops, recursion, or dynamic programming,
* tasks where verbose explanation may break code extraction,
* adversarial prompts with ambiguous wording.

### 8.5 Evaluation Metrics

| Metric                       | Why It Matters                                             |
| ---------------------------- | ---------------------------------------------------------- |
| pass_at_1                    | Measures complete task success under single generation     |
| syntax_pass_rate             | Detects basic code validity                                |
| avg_test_pass_rate           | Captures partial correctness across unit tests             |
| exact_match                  | Provides a strict but secondary text-level signal          |
| avg_code_token_f1            | Measures similarity while tolerating alternative solutions |
| code extraction success rate | Measures output-format reliability                         |
| function signature accuracy  | Critical for MBPP-style tests                              |
| average generation length    | Detects verbosity or truncation                            |
| average inference latency    | Supports practical deployment comparison                   |
| peak GPU memory              | Necessary for resource-aware conclusions                   |
| trainable parameter ratio    | Distinguishes full tuning from PEFT methods                |
| data valid-pair ratio        | Shows A3 dataset quality                                   |
| pair separability score      | Shows whether preference labels are meaningful             |

## 9. Risk Analysis and Mitigation

| Risk                                | Impact                                      | Mitigation                                                      |
| ----------------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| Online model download unavailable   | Runtime failure                             | Use local cached Qwen3 and offline environment variables        |
| Qwen3 template mismatch             | Poor generation despite successful training | Verify tokenizer and template behavior during preflight         |
| vGPU or memory instability          | Full training may fail                      | Use QLoRA/QDoRA fallback and mini-step smoke tests              |
| DPO data noise                      | Alignment may learn artifacts               | Use A3 quality scorecard, bad-pair filtering, and ablation sets |
| Length bias in preference data      | Model may prefer verbose answers            | Track chosen/rejected length gaps and compare filtered variants |
| Evaluation path inconsistency       | Metrics may be invalid or unreproducible    | Standardize MBPP path and metric output contract                |
| Old Qwen1.5 outputs pollute results | Confusing final report                      | Use Qwen3-prefixed output directories and explicit run IDs      |
| Overfitting to small test samples   | Inflated results                            | Maintain held-out generation set and report error categories    |

## 10. Expected Deliverables

### 10.1 A2 Deliverables

* Qwen3 base validation record.
* Full SFT configuration and trained checkpoint if feasible.
* QLoRA and QDoRA experimental configurations.
* Prediction outputs for fixed test prompts.
* MBPP-style execution metrics and error cases.
* Training logs, loss curves, memory records, and model-selection notes.
* README explaining independent execution, input/output contract, and integration path.

### 10.2 A3 Deliverables

* Cleaned DPO ranking dataset.
* Generation-format held-out test data.
* Dataset registry file.
* Data statistics report.
* Preference quality report.
* Sample preview and bad-pair examples.
* README explaining the dataset contract and downstream usage.

### 10.3 Integration Deliverables

* A2 model path for A4 policy initialization.
* A3 dataset path for A4 alignment training.
* Metrics and cases files for A6 unified comparison.
* Demonstration screenshots or videos for independent module inspection.
* Final module report with ablation tables and failure analysis.

## 11. Why This Proposal Can Stand Out

This plan goes beyond completing the minimum SFT and data-conversion tasks. It treats A2 and A3 as engineering-critical modules in a modern LLM optimization system.

The key differentiators are:

1. A full-system Qwen3 migration instead of merely following the default Qwen1.5 path.
2. A resource-aware training ladder from Full SFT to QLoRA and QDoRA.
3. A preference-data quality scorecard that makes A3 independently valuable.
4. A data-contract design that supports DPO, SimPO, ORPO, KTO, and verifier-based extensions.
5. Execution-aware evaluation, not only text similarity.
6. Reproducible artifact naming, metrics, logs, and risk mitigation.

The expected final result is not just a trained model and a converted dataset, but a defensible, extensible, and auditable fine-tuning subsystem.

## 12. References

[1] Qwen Team. *Qwen3-0.6B Model Card*. Hugging Face. [https://huggingface.co/Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
[2] Qwen Team. *Qwen3 Technical Report*. arXiv:2505.09388. [https://arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)
[3] hiyouga / LLaMA-Factory Team. *LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs*. GitHub. [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
[4] Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., & Ma, Y. *LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models*. arXiv:2403.13372. [https://arxiv.org/abs/2403.13372](https://arxiv.org/abs/2403.13372)
[5] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
[6] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314. [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
[7] Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353. [https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)
[8] NVIDIA Research. *DoRA: Weight-Decomposed Low-Rank Adaptation*. GitHub. [https://github.com/NVlabs/DoRA](https://github.com/NVlabs/DoRA)
[9] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290. [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
[10] Hugging Face. *PEFT LoRA / DoRA Documentation*. [https://huggingface.co/docs/peft/main/package_reference/lora](https://huggingface.co/docs/peft/main/package_reference/lora)
[11] Hugging Face. *Transformers Documentation*. [https://huggingface.co/docs/transformers/index](https://huggingface.co/docs/transformers/index)
[12] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., et al. *HuggingFace’s Transformers: State-of-the-art Natural Language Processing*. arXiv:1910.03771. [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771)
[13] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. arXiv:1912.01703. [https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)
[14] PyTorch Foundation. *PyTorch Official Documentation*. [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
[15] Hugging Face. *bitsandbytes Quantization Documentation*. [https://huggingface.co/docs/bitsandbytes/index](https://huggingface.co/docs/bitsandbytes/index)
[16] Jon Durbin. *py-dpo-v0.1 Dataset*. Hugging Face Datasets. [https://huggingface.co/datasets/jondurbin/py-dpo-v0.1](https://huggingface.co/datasets/jondurbin/py-dpo-v0.1)
[17] Google Research. *MBPP: Mostly Basic Python Problems*. GitHub. [https://github.com/google-research/google-research/tree/master/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
[18] Hugging Face. *Datasets Documentation*. [https://huggingface.co/docs/datasets/index](https://huggingface.co/docs/datasets/index)
[19] Hugging Face. *Accelerate Documentation*. [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)
[20] Apache Software Foundation. *Apache Parquet Documentation*. [https://parquet.apache.org/docs/](https://parquet.apache.org/docs/)