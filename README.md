# BEYOND LIKELIHOOD：执行引导的 Python 代码生成系统

> 项目路径：`/root/project`  
> 报告依据：服务器截至 2026-07-14 的源码、配置、运行清单和已落盘指标  
> 技术主线：Curate · Adapt · Rank · Align · Verify

---

## 团队与分工

| 角色 | 姓名 | 学号 | 负责模块 |
|---|---|---|---|
| 组长 | 冯隆腾 | 20235953 | A4 偏好对齐、A5 推理增强与 PathCoder |
| 组员 | 苏焜 | 20235805 | A2 监督微调、A3 偏好数据构造与评分 |
| 组员 | 牛鹏浩 | 20235775 | A1 数据治理与预处理 |

## GitHub 源码发布说明

本仓库保存 `/root/project` 中的项目代码、脚本、配置、文档和可合理纳入 Git
版本控制的轻量数据。受 GitHub 单文件与仓库体积限制，仓库不包含基础模型、
训练后模型权重、checkpoint、优化器状态、缓存、日志及大型运行输出。相关运行
路径和复现实验命令仍保留在脚本与文档中；使用前需自行准备 `Qwen3-0.6B`
模型并按文档配置环境。

---

## 1. 项目概述

### 1.1 项目名称

`BEYOND LIKELIHOOD：基于 Qwen3-0.6B 的执行引导 Python 代码生成系统`

### 1.2 项目目标

本项目面向 Python 代码生成任务，解决“模型输出看起来合理，但语法错误、接口不匹配或无法通过测试”的问题。系统以 Qwen3-0.6B 为基础模型，构建从监督数据治理、资源受限微调、偏好数据审计、DPO/PPO 对齐，到测试时多路径搜索与受限执行验证的完整链路。

项目不只优化语言模型的 token 似然，还把 Python AST、单元测试结果、偏好对和运行时反馈纳入训练或候选选择。最终系统能够：

1. 从原始代码指令数据构造可复现、可审计的训练集；
2. 在约 23 GiB 可见显存上完成 Full SFT、QLoRA、QDoRA 对照实验；
3. 构造并筛选 DPO 高置信偏好对；
4. 进行 LoRA-DPO 与奖励函数 PPO 对齐实验；
5. 使用 Greedy、CoT、Self-Consistency、Best-of-N、Reflexion、Tree of Thoughts 和 Hybrid 等推理策略；
6. 在隔离子进程中执行生成代码，以 MBPP Sanitized 的测试通过率衡量真实功能正确性。

### 1.3 当前完成情况

| 类型 | 完成情况 |
|---|---|
| 基础要求 | A1 数据构造、A2 SFT、A3 偏好数据、A4 DPO、A5 代码生成与执行评测均已实现并有落盘产物 |
| 进阶要求 | Full/QLoRA/QDoRA 对照；高置信偏好评分；LoRA-DPO；奖励函数 PPO；Self-Consistency、Best-of-N、Reflexion、ToT、Hybrid；PathCoder 统一演示入口 |
| 支持的主要任务类型 | MBPP 风格 Python 函数生成、带 `assert` 测试的自定义编程任务、模型/策略批量评测 |
| 当前限制 | 不同历史评测协议不可直接比较；集成 A4-DPO 未超过 A2 Full；ToT 成本高且收益有限；项目尚无完整环境锁文件 |

---

## 2. 整体流程与模块结构

### 2.1 模块边界

| 模块 / 阶段 | 入口文件 / 入口函数 | 主要职责 | 输入 | 输出 |
|---|---|---|---|---|
| A1 数据治理 | `sft/scripts/prepare_data.sh`、`prepare_code_sft_data.py` | 字段归一化、空值与长度过滤、去重、AST 语法检查、固定随机种子切分、坏样本审计 | `python_code_instructions_18k_alpaca` Parquet | `code_sft_{train,valid,test}.json`、统计和坏样本 |
| A2 监督微调 | `sft/scripts/run_a2_v2.sh` | Full SFT、QLoRA、QDoRA 训练、checkpoint 选择、合并、评测和 A4 桥接 | A1 JSON、Qwen3-0.6B | Full 模型或 PEFT adapter、交接清单、比较报告 |
| A3 偏好数据 | `dpo/scripts/run_a3_v2.sh`、`dpo/a3_v2/pipeline.py` | 清洗 chosen/rejected 对、质量打分、分层数据视图、A4 一步探针 | `py-dpo-v0.1/py-dpo.parquet` | raw/clean/high-confidence/debug 偏好集和质量报告 |
| A4 偏好对齐 | `best_path/chain.py::render_a4`、`best_path/scripts/run_full_chain.sh` | 从 A2 Full 进行 LoRA-DPO；导出合并模型；补充奖励函数 PPO 与质量守卫实验 | A2 模型、A3 高置信偏好对 | DPO adapter、合并模型、PPO adapter、训练与 MBPP 指标 |
| A5 测试时增强 | `best_path/evaluate.py`、`tts/test_time_scaling.py` | 多候选生成、AST 归一化、一致性投票、执行重排、错误反馈修复 | 模型路径、任务、可见测试 | 最终代码、每题轨迹、策略指标和比较报告 |
| 统一演示 | `pathcoder`、`tts/showcase_cli.py` | 环境诊断、示例选择、七种策略、自定义任务、历史会话和指标目录 | CLI 参数或交互输入 | 终端结果、`showcase_sessions/*.json` |

### 2.2 系统架构图

```mermaid
flowchart LR
    R1["18,612 条代码指令"] --> A1["A1 清洗与冻结切分"]
    A1 --> D1["9,294 / 516 / 517"]
    D1 --> A2["A2 Full / QLoRA / QDoRA"]

    R2["9,466 条原始偏好数据"] --> A3["A3 质量评分与筛选"]
    A3 --> D3["8,797 条高置信偏好对"]

    A2 --> A4["A4 LoRA-DPO / PPO"]
    D3 --> A4
    A4 --> M4["Adapter 与合并模型"]

    A2 --> A5["A5 多路径推理"]
    M4 --> A5
    A5 --> EX["代码提取 → AST → 受限执行"]
    EX --> OUT["最终代码、轨迹与 MBPP 指标"]
```

### 2.3 一次完整实验的流程

1. A1 从 `python_code_instructions_18k_alpaca` 读取 Parquet，统一为 Alpaca 风格 `instruction/input/output`，执行规则过滤、去重与 AST 校验，并使用 seed 42 固定切分。
2. A2 在同一数据和 Qwen3 模板下训练 Full、QLoRA、QDoRA；训练配置统一使用 `bf16: true`、`fp16: false`、`template: qwen3`、`enable_thinking: false`。
3. A2 对 epoch checkpoint 使用固定 MBPP 子集选择，并输出模型路径、adapter 路径、合并路径及哈希清单。
4. A3 将原始 `prompt/chosen/rejected` 转为 LLaMA-Factory ranking 格式，根据完整性、chosen 代码质量、可分离性、长度安全、长度平衡和格式计算质量分。
5. A4 读取 A2 Full 与 A3 high-confidence 数据，进行 LoRA-DPO；实验分支还可启动本地奖励 API，以执行测试、语法、安全性、接口完整性和反投机规则训练 PPO adapter。
6. A5 对 A2、A4 或 PPO checkpoint 生成一个或多个候选，提取 Python 代码，进行 AST 校验，并在带 CPU、内存、文件大小和超时限制的临时目录中执行测试。
7. 系统按测试通过数、完整任务通过、语法、一致性和格式选择最终结果；失败样本可转成 Reflexion 修复提示。
8. 每个版本化运行目录保存 manifest、配置、模型、日志、GPU 遥测、逐题 cases、指标 JSON/CSV/Markdown 和训练曲线。

---

## 3. 模型、数据集与外部资源

### 3.1 模型说明

| 项目 | 内容 |
|---|---|
| 使用模型 | Qwen3-0.6B，约 0.6B 参数、28 层、32K 原生上下文 |
| 模型来源 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，Apache-2.0 |
| 项目内相对路径 | `Qwen3-0.6B/` |
| 主要派生模型 | `sft/outputs/a2_v2_runs/a2_formal_001/`；`best_path/runs/best_full_dpo_a5_20260702/models/` |
| 是否需要 GPU | 训练需要；正式批量推理建议使用 GPU；部分数据处理和单元测试可在 CPU 上运行 |
| 是否需要联网运行 | 不需要；正式脚本设置 `HF_DATASETS_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` |

服务器已存在完整模型，不需要重复下载。新环境可准备到以下目录：

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir /root/project/Qwen3-0.6B
```

### 3.2 数据集 / 示例数据说明

| 数据或文件 | 用途 | 来源 | 项目内相对路径 |
|---|---|---|---|
| Python Code Instructions 18K | A1 原始 SFT 指令数据 | [sahil2801/code_instructions_120k](https://huggingface.co/datasets/sahil2801/code_instructions_120k) 的 18K 子集 | `python_code_instructions_18k_alpaca/` |
| A1 冻结切分 | A2 训练、验证、测试 | A1 自行构造 | `sft/data/code_sft_{train,valid,test}.json` |
| py-dpo-v0.1 | A3 原始偏好数据；chosen 源自 Tested-22k-Python-Alpaca，CC-BY-4.0 | 项目本地数据卡 | `py-dpo-v0.1/py-dpo.parquet` |
| A3 high-confidence | A4 DPO 正式输入 | A3 质量评分与阈值筛选 | `dpo/data/a3_v2_runs/a3_v2_verify_20260702_005837/code_dpo_a3_v2_high_confidence_train.json` |
| MBPP Sanitized | 独立代码生成与执行评测 | [Google Research MBPP](https://github.com/google-research/google-research/tree/master/mbpp) | `/root/mbpp/sanitized/test-00000-of-00001.parquet` |
| PathCoder 内置样例 | 现场快速演示 | 项目自带 | `tts/showcase_cli.py` 中的示例目录 |

A1 实际数据统计：原始 18,612 条，保留 10,327 条；训练/验证/测试分别为 9,294/516/517。A3 实际统计：原始 9,466 条，high-confidence 训练集 8,797 条，生成测试集 485 条。

---

## 4. 环境安装

### 4.1 运行环境

| 项目 | 已验证环境 |
|---|---|
| Python 版本 | Python 3.11.15 |
| 操作系统 / 服务器环境 | Linux 服务器；项目根目录 `/root/project`；Conda 环境 `/opt/conda/envs/shixun` |
| GPU | NVIDIA H200 NVL，单分区可见显存 23,552 MiB，驱动 580.119.02 |
| 深度学习栈 | PyTorch 2.12.1、Transformers 5.6.0、PEFT 0.18.1、TRL 0.9.6 |
| 数据与加速依赖 | bitsandbytes 0.49.2、Accelerate 1.11.0、Datasets 4.0.0、PyArrow 24.0.0 |
| 训练框架 | 项目内 `LlamaFactory/`，通过兼容入口调用 |

### 4.2 安装与检查步骤

提供的服务器已配置环境，推荐按以下方式启动：

```bash
cd /root/project
source /opt/conda/etc/profile.d/conda.sh
conda activate shixun

# 检查模型、数据、Python 依赖、GPU 和 Qwen3 模板
bash scripts/runtime_check.sh
```

如需在新服务器重建环境，应使用 Python 3.11、匹配 CUDA 13 的 PyTorch/bitsandbytes，并安装上表中的已验证版本。当前仓库没有完整的 Conda lockfile，因此应先完成运行时检查和 1-step smoke test，再提交正式训练。

常见环境问题：

- Qwen3 数据不包含 thinking target，正式训练统一设置 `enable_thinking: false`。
- QLoRA/QDoRA 需要 CUDA 13 动态库和 `BNB_CUDA_VERSION=130`；相关设置在 `sft/a2_v2/common_env.sh`。
- Transformers 5.6.0 与 PEFT 0.18.1 的 adapter 自动加载存在私有 API 不兼容；`tts/test_time_scaling.py` 已改为显式加载基座模型和 `PeftModel`。
- 使用历史 LLaMA-Factory 配置时，应通过 `python -m sft.a2_v2.llamafactory_compat`，不要直接修改 vendor 目录。
- 正式训练会产生多个 checkpoint，应提前检查磁盘空间并使用新的 `RUN_ID`，避免覆盖历史证据。

---

## 5. 输入文件与配置文件说明

### 5.1 主要配置文件

| 配置文件 | 作用 | 需要关注的字段 |
|---|---|---|
| `sft/configs/a2_v2/full_formal.yaml` | Full SFT 正式配置 | `model_name_or_path`、`dataset`、batch、epoch、learning rate |
| `sft/configs/a2_v2/qlora_formal.yaml` | 4-bit QLoRA 正式配置 | quantization、LoRA rank/target、batch |
| `sft/configs/a2_v2/qdora_formal.yaml` | 4-bit QDoRA 正式配置 | `use_dora`、LoRA 参数、量化参数 |
| `dpo/configs/qwen3_06b_code_lora_dpo.yaml` | 基础 LoRA-DPO | base model、ranking dataset、`pref_beta`、LoRA 参数 |
| `best_path/runs/<RUN_ID>/configs/a4_lora_dpo.yaml` | 集成 A4 LoRA-DPO | A2 模型、A3 数据、rank 16、alpha 32、2 epochs |
| `dpo/configs/a2full_best_ppo_reward_fn.yaml` | A2 Full 奖励函数 PPO 模板 | PPO batch、KL、LoRA、reward API、输出目录 |
| `tts/run_test_time_scaling.sh` | A5 多策略统一启动 | `MODEL_PATH`、`STRATEGY`、候选数、输出目录 |

### 5.2 主要输入文件

| 输入文件 | 用途 | 适用场景 |
|---|---|---|
| `sft/data/code_sft_train.json` | Qwen3 代码指令学习 | A2 训练 |
| `sft/data/code_sft_valid.json` | epoch 级损失与模型选择辅助 | A2 验证 |
| `sft/data/code_sft_test.json` | 冻结 held-out 生成质量评测 | A2 评测 |
| `dpo/data/a3_v2_runs/<RUN_ID>/dataset_info.json` | 注册 ranking 数据字段映射 | A3→A4 联调 |
| `.../code_dpo_a3_v2_high_confidence_train.json` | 高置信 chosen/rejected 对 | A4 LoRA-DPO |
| `dpo/data/a2full_reward_ppo_train.json` | 奖励函数 PPO prompts | A4 PPO 扩展 |
| `/root/mbpp/sanitized/test-00000-of-00001.parquet` | 257 个任务、776 条测试 | A2/A4/A5 统一执行评测 |
| CLI `--task` 与重复 `--test` | 自定义任务和断言 | PathCoder 模块演示 |

---

## 6. 完整流程 Demo 运行

### 6.1 Demo 样例说明

| Demo | 输入文件 / 输入内容 | 演示目的 |
|---|---|---|
| PathCoder 快速演示 | 内置 `two_sum` | 验证环境、模型加载、代码生成和执行测试 |
| PathCoder 综合演示 | 内置 `merge_intervals`，`hybrid` 策略 | 展示多路径搜索、重排和错误修复 |
| 完整 A1→A5 实验 | A1 冻结数据、A3 high-confidence、MBPP | 重新训练 A2/A4 并执行 A5 |
| A5 全策略比较 | MBPP 257 题 | 比较 Greedy、CoT、SC、BoN、Reflexion、ToT |

### 6.2 运行命令

快速演示：

```bash
pathcoder doctor
pathcoder run --example two_sum --strategy greedy
pathcoder run --example merge_intervals --strategy hybrid
```

自定义任务：

```bash
pathcoder run \
  --strategy hybrid \
  --task "Write a function factorial(n) that raises ValueError for n < 0." \
  --test "assert factorial(0) == 1" \
  --test "assert factorial(5) == 120"
```

完整训练链会运行较长时间，必须使用新的运行 ID：

```bash
cd /root/project
conda activate shixun

RUN_ID=team_full_$(date +%Y%m%d_%H%M%S) \
A3_RUN_ID=a3_v2_verify_20260702_005837 \
bash best_path/scripts/submit_full_chain.sh all
```

A5 全策略比较：

```bash
MODEL_PATH=/root/project/best_path/runs/best_full_dpo_a5_20260702/models/a2_full \
STRATEGY=compare NUM_SAMPLES=5 BEST_OF_N=5 REFLEXION_ROUNDS=2 \
TREE_WIDTH=3 TREE_DEPTH=2 \
OUTPUT_DIR=/root/project/tts/outputs/tts_compare_$(date +%Y%m%d_%H%M%S) \
bash tts/run_test_time_scaling.sh
```

### 6.3 关键参数说明

| 参数 | 说明 |
|---|---|
| `RUN_ID` | 隔离数据、配置、模型、日志与报告；不要复用正式历史运行 ID |
| `A3_RUN_ID` | 指定 A4 使用的 A3 偏好数据版本 |
| `MODEL_PATH` | A2 Full、A4 合并模型或 PEFT adapter checkpoint |
| `STRATEGY` | `greedy/cot/self_consistency/best_of_n/reflexion/tot/compare` |
| `NUM_SAMPLES` / `BEST_OF_N` | Self-Consistency 或 Best-of-N 候选数；越大越慢 |
| `REFLEXION_ROUNDS` | 最大错误反馈修复轮数 |
| `TREE_WIDTH` / `TREE_DEPTH` | ToT 搜索宽度和深度 |
| `OUTPUT_DIR` | A5 cases、metrics 和 comparison 的独立输出根目录 |
| `--limit` | 仅运行前 N 条任务，用于 smoke test；正式结果应为 257 条 |

### 6.4 运行成功的判断方式

- `bash scripts/runtime_check.sh` 末尾输出 `runtime_check=PASS`。
- 完整链日志出现 `Best-path command completed`，且 `<RUN_DIR>/reports/final_report.json` 存在。
- A5 日志显示 `Wrote reports to ...`，每个策略目录包含 `metrics.json` 和 `cases.jsonl`。
- 正式 MBPP 指标中的 `total` 为 257，`total_tests` 为 776。
- 自定义 PathCoder 任务显示最终代码、测试结果和会话保存路径。

---

## 7. 输出文件与结果说明

### 7.1 主要输出文件

| 输出文件 | 生成模块 / 阶段 | 格式 | 说明 |
|---|---|---|---|
| `sft/data/data_statistics.json` | A1 | JSON | 原始/保留数量、过滤原因、长度分布和 seed |
| `sft/data/bad_cases.json` | A1 | JSON | 被过滤样本及原因，用于审计 |
| `sft/outputs/a2_v2_runs/<RUN_ID>/a2_handoff_manifest.json` | A2 | JSON | Full/adapter/merged 路径和训练契约 |
| `.../reports/comparison_report.{json,csv,md}` | A2 | 多格式 | Full、QLoRA、QDoRA 的训练、显存和评测对照 |
| `dpo/data/a3_v2_runs/<RUN_ID>/preference_quality_report.json` | A3 | JSON | 偏好质量、相似度、长度、语法代理和过滤统计 |
| `best_path/runs/<RUN_ID>/models/a4_dpo_adapter/` | A4 | PEFT 权重 | LoRA-DPO adapter 与训练曲线 |
| `best_path/runs/<RUN_ID>/models/a4_dpo_merged/` | A4 | 模型权重 | 与 A2 基座合并后的推理模型 |
| `dpo/outputs/<RUN_ID>/ppo_guard_report.json` | A4 PPO | JSON | PPO 相对起点的退化阈值和质量守卫结论 |
| `best_path/runs/<RUN_ID>/eval/*/metrics.json` | A4/A5 | JSON | 同一集成协议下的 greedy/enhanced 指标 |
| `tts/outputs/<RUN_ID>/<strategy>/cases.jsonl` | A5 | JSONL | 每题候选、代码、测试结果、选择细节 |
| `tts/outputs/<RUN_ID>/comparison.{json,csv,md}` | A5 | 多格式 | 六种批量策略的效果、耗时和显存对比 |
| `tts/outputs/showcase_sessions/*.json` | PathCoder | JSON | 自定义演示的完整可复现会话 |

### 7.2 已落盘结果摘要

#### 数据与训练

| 阶段 | 关键结果 |
|---|---|
| A1 | 18,612 → 10,327；去除 8,285 条噪声；固定切分 9,294/516/517 |
| A2 formal Full | 596.05M 参数全量训练；MBPP pass@1 6.23%；峰值分配显存约 9.79 GiB |
| A2 formal QLoRA | 5.05M 可训练参数（0.8395%）；MBPP pass@1 3.50%；峰值约 2.58 GiB |
| A2 formal QDoRA | 5.39M 可训练参数（0.8962%）；MBPP pass@1 4.28%；峰值约 2.59 GiB |
| A3 | 8,924 clean 偏好对中 8,797 条达到 0.72 高置信阈值；平均质量分 0.9192 |
| A4 集成 DPO | LoRA rank 16、2 epochs；train loss 0.0623；正式输出包含 adapter 和 merged model |

> 数据版本说明：`a2_formal_001/a2_handoff_manifest.json` 记录的是较早的 A1 快照（16,348/908/909），而最终 `best_full_dpo_a5_20260702/manifest.json` 使用当前冻结快照（9,294/516/517）。因此上表 A2 formal 适合比较三种微调方法的资源与相对表现，最终 A1→A5 结论以 best-path manifest 和同协议评测为准。

#### A4/A5 同协议结果

`best_path/runs/best_full_dpo_a5_20260702/reports/final_report.json` 使用“可见 MBPP 测试同时进入提示词和执行验证器”的集成协议：

| 路线 | 策略 | pass@1 | 语法通过率 | 平均测试通过率 | 平均耗时/题 |
|---|---:|---:|---:|---:|---:|
| A2 Full | Greedy | 49.81% | 99.61% | 55.54% | 0.97 s |
| A4 DPO epoch 2 | Greedy | 21.40% | 84.82% | 23.97% | 4.39 s |
| A2 Full → A5 | Best-of-8 + 执行重排 + Reflexion | **65.37%** | **100%** | **71.52%** | 7.25 s |
| A4 DPO → A5 | 同上 | 50.58% | **100%** | 55.41% | 26.62 s |

该结果表明：当前 A4 DPO 在集成协议下发生退化，因此正式质量门同时评测 A2 和 A4 两条路线，最终最佳质量来自 A2 Full → A5，而不是强制使用 A4 输出。

#### A5 六策略同模型比较

`tts/outputs/tts_a2_full_compare/comparison.json` 使用同一 A2 Full、同一 MBPP 数据和执行器：

| 策略 | pass@1 | 语法通过率 | 平均耗时/题 | 平均生成序列/题 |
|---|---:|---:|---:|---:|
| Greedy | 46.69% | 99.22% | 1.86 s | 1.0 |
| CoT | 40.08% | 88.72% | 3.58 s | 1.0 |
| Self-Consistency（5） | **63.04%** | **100%** | 18.21 s | 5.0 |
| Best-of-N（5） | 60.31% | **100%** | 18.76 s | 5.0 |
| Reflexion（2） | 45.14% | 96.89% | 9.86 s | 2.17 |
| ToT（3×2） | 43.58% | 94.94% | 40.32 s | 15.0 |

此外，PPO `checkpoint-2800` 经过 5-sample Self-Consistency 后达到 pass@1 59.53%、语法通过率 100%，结果位于 `tts/outputs/tts_sc_ppo_reward_epoch_continue_20260714/`。

> 评测口径说明：A2 formal、基础 DPO、best_path 和 TTS 历史运行使用过不同提示模板或是否向模型展示测试。报告仅在明确标注“同协议”的表格内做直接比较。

### 7.3 结果图与可视化位置

项目已保存以下可视化，可直接用于答辩截图：

- A2 方法对比：`sft/outputs/a2_v2_runs/a2_formal_001/reports/comparison_metrics.png`
- A3 过滤漏斗：`dpo/data/a3_v2_runs/a3_v2_verify_20260702_005837/reports/filter_funnel.png`
- A3 质量分布：同目录的 `quality_score_hist.png`、`pair_similarity_hist.png`
- A4 DPO 曲线：`best_path/runs/best_full_dpo_a5_20260702/models/a4_dpo_adapter/training_rewards_accuracies.png`
- A1/A4 训练曲线：各模型目录下的 `training_loss.png` 和 `training_eval_loss.png`
- A5 策略表：`tts/outputs/tts_a2_full_compare/comparison.md`

---

## 8. 协作实现说明

团队按 A1–A5 划分职责，但通过版本化契约而不是人工拷贝临时文件联调：

- A1 输出固定字段的 Alpaca JSON、统计报告、坏样本和 seed 42 切分；下游将其视为只读输入。
- A2 使用 `a2_handoff_manifest.json` 声明 Full、adapter 和 merged 路径，并记录 `template/enable_thinking/cutoff_len`。
- A3 为 raw、clean、high-confidence、debug 和 generation-test 建立独立数据视图，通过 `dataset_info.json` 注册 chosen/rejected 字段。
- A2 和 A3 都提供 1-step A4 bridge，提前验证 LLaMA-Factory DPO 能否加载模型与 ranking 数据。
- `best_path/chain.py` 在新运行目录中复制输入快照、计算 SHA-256，并生成 A2/A4 配置，避免上游数据变化污染正式结果。
- A4 同时保留 adapter 和合并模型，A5 可按统一 `MODEL_PATH` 接口消费 A2、A4 或 PPO checkpoint。
- A5 把每个候选、执行反馈和选择理由写入 JSONL，使团队可以复查单题，而不只查看汇总分数。
- 所有正式输出使用 `RUN_ID` 隔离，stage marker 支持安全续跑；日志与 GPU telemetry 与模型产物保存在同一运行根目录。

---

## 9. 已知问题与改进方向

| 问题 | 当前原因 | 可能改进 |
|---|---|---|
| 集成 A4-DPO 低于 A2 Full | 偏好分布、DPO 学习率/epoch、提示协议和 A2 强基座之间存在失配 | 降低学习率与 beta，加入 held-out 执行选择和早停；将 A4 是否晋级设为质量门而非固定步骤 |
| PPO v2 质量守卫未通过 | PPO 三项指标均小幅下降，`any_metric_improved=false` | 增加离线 reward 校准、缩小 KL 漂移、按执行收益选择 checkpoint，拒绝未增益产物 |
| A3 train/test 存在 237 个唯一 prompt hash 重叠 | 当前项目策略允许 prompt overlap，只报告不删除 | 增加严格去重 split，发布 overlap-free 指标并和现有口径并列 |
| 多套历史评测不可直接比较 | prompt、代码提取、是否展示测试、采样参数不同 | 建立单一 protocol manifest，指标中记录 prompt hash、decoder 参数和 evaluator 版本 |
| A2 formal 与最终 best-path 使用不同 A1 快照 | 项目演进期间数据重新治理，历史运行仍被保留 | 报告中强制展示数据 SHA-256/行数；后续正式实验只从冻结 manifest 启动 |
| Self-Consistency 成本约为 Greedy 的 10 倍 | 每题生成 5 条长 CoT 路径 | 对简单题早停；按置信度自适应采样；缓存重复 prompt；使用轻量 verifier |
| Self-Consistency 平均多数一致率仅 0.2117 | 五个候选高度多样，几乎没有完全一致 | 研究语义等价归一化、行为签名投票和测试覆盖引导，而不只依赖 AST 结构 |
| ToT 更慢且未超过 Greedy | 当前启发式 thought score 与真实代码正确性相关性弱 | 训练/校准过程奖励模型，动态剪枝，减少无效思路扩展 |
| 执行沙箱属于基础资源限制 | 临时目录与 rlimit 不能替代容器级隔离 | 使用无网络容器、seccomp、只读文件系统与更严格系统调用白名单 |
| 环境复现依赖服务器快照 | 尚无完整 Conda lockfile / 容器镜像 | 导出 `environment.yml`、pip lock 与 CUDA 镜像，并纳入 CI smoke test |
| 当前模型规模较小 | Qwen3-0.6B 容量限制复杂算法与长程序 | 在同一数据和协议下扩展到更大 Qwen3 模型，并保留 0.6B 作为成本基线 |
