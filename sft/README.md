# Qwen3-0.6B Python Code SFT

本目录用于在 `/root/project` 工作区内，用 LLaMA-Factory 对本地 `Qwen3-0.6B` 做 Python 代码指令 SFT。

所有运行脚本都会先切到工作区根目录：

```bash
cd /root/project
```

因此配置中的模型、数据和输出路径都按 `/root/project` 的相对路径写法设置，例如 `./Qwen3-0.6B`、`./sft/data`、`./sft/outputs/...`。

## 目录

- `configs/qwen3_06b_code_full_sft.yaml`：全量 SFT 配置。
- `configs/qwen3_06b_code_full_predict.yaml`：测试集批量生成配置。
- `scripts/prepare_code_sft_data.py`：数据处理模块，把 parquet 转成 LLaMA-Factory Alpaca JSON，并划分 train/valid/test。
- `scripts/infer.py`：单条或批量推理脚本。
- `scripts/evaluate_code_predictions.py`：MBPP sanitized 执行评测脚本，会实际运行候选代码，并按测试样例通过率输出 `pass_at_1`、`syntax_pass_rate`、`avg_test_pass_rate`。

## 运行流程

准备数据：

```bash
bash sft/scripts/prepare_data.sh
```

训练：

```bash
bash sft/scripts/train.sh
```

在 held-out test 上批量生成：

```bash
MODEL_PATH=/root/project/sft/outputs/qwen3_06b_code_full_sft/checkpoint-600 \
bash sft/scripts/predict_full.sh
```

评测代码生成结果：

```bash
bash sft/scripts/evaluate_full.sh
```

单条推理：

```bash
bash sft/scripts/infer_full.sh --prompt "Write a Python function to check whether a string is a palindrome."
```

一键跑完整流程：

```bash
bash sft/scripts/run_all.sh
```

## 常用参数

快速小样本调试：

```bash
LIMIT=1000 bash sft/scripts/prepare_data.sh
```

指定 GPU：

```bash
GPU_ID=0 bash sft/scripts/train.sh
```

使用 bf16 时，手动把 `configs/qwen3_06b_code_full_sft.yaml` 里的 `fp16: false`、`bf16: true` 打开即可。

## 输出

- 处理后的数据：`sft/data/code_sft_train.json`、`sft/data/code_sft_valid.json`、`sft/data/code_sft_test.json`、`sft/data/mbpp_sanitized_test.json`
- 微调模型：`sft/outputs/qwen3_06b_code_full_sft`
- 测试集生成：`sft/outputs/qwen3_06b_code_full_predict/generated_predictions.jsonl`
- 代码评测：`sft/outputs/eval_mbpp/mbpp_metrics.json`
