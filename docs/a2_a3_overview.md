# A2 / A3 个人模块验收

> 项目：Qwen3-0.6B Python 代码生成优化系统  
> 个人负责：A2 监督微调、A3 偏好数据构造  
> 工作目录：`/root/project`  
> 运行环境：Conda `shixun`，NVIDIA H200 NVL

## 1. 概览

A2 的基础功能是用 Qwen3-0.6B 完成 Full SFT。我的增量工作是把它扩展为 Base、Full、QLoRA、QDoRA 四条可对照路线，并补全环境预检、CUDA 13 bitsandbytes 兼容、固定随机种子、独立 RUN_ID、显存与速度采集、checkpoint 选择、adapter 合并、统一评测和 A4 桥接。正式实验中，Full 在当前 A2 同协议 MBPP 上最好，pass@1 为 16/257；QLoRA 和 QDoRA 分别为 9/257、11/257。QLoRA/QDoRA 只训练约 0.84%/0.90% 参数，正式运行的 PyTorch peak allocated 约 2.58/2.59 GiB，而 Full 约 9.79 GiB。这个结果说明 PEFT 的主要价值是显存和可训练参数效率，不保证墙钟时间更短；本机上 QDoRA 反而最慢。

A3 的基础功能是把 Parquet 中的 prompt、chosen、rejected 转成 LLaMA-Factory ranking 数据。我的增量工作是加入硬无效检查、去重、代码抽取、Python AST 语法检查、Qwen3 tokenizer 长度统计、chosen/rejected 相似度和长度偏差分析，再以六维可解释质量分生成 raw、clean、high-confidence、debug 和 generation-test 五种视图，同时输出逐样本审计、坏样本、统计报告和图表。已验证 run 从 9,466 条源数据得到 8,924 条 raw train、8,797 条 high-confidence train、128 条 debug 和 485 条 generation test；high-confidence 相对 raw 进一步隔离了 127 条低分 pair。这个分数是静态质量代理，不等价于程序正确率，最终是否改善 DPO 必须依赖同协议消融和执行测试。

集成方面，我没有改坏或覆盖老师给的 default：A2/A3 v2 都使用新增目录和 RUN_ID。A2 会输出 handoff manifest，A3 会输出 ranking `dataset_info.json`；桥接脚本已经证明 Full、QLoRA merged、QDoRA merged 都能读取 A3 high-confidence 数据并完成 1-step DPO update。Qwen3 的训练与验证均采用 `template: qwen3`、`enable_thinking: false`，因为现有监督和偏好数据没有规范的 thinking 样本。

## 2. 模块关系

```text
A1 监督数据 ──> A2：Full / QLoRA / QDoRA SFT ──┐
                                              ├──> A4：偏好对齐 ──> A5：推理增强
原始偏好数据 ─> A3：清洗、评分、分层、审计 ──────┘
```

## 3. 个人成果

| 模块 | Default 能力 | 基础成果 | 进阶成果 |
|---|---|---|---|
| A2 | 单一 Full SFT 脚本 | Qwen3 Full SFT、保存与预测 | QLoRA、QDoRA、资源采集、checkpoint 选择、adapter 合并、RUN_ID 隔离 |
| A3 | Parquet 映射、shuffle、基础去重、train/test JSON | LLaMA-Factory ranking 与 generation 数据 | 六维质量分、AST/相似度/token 长度、五种数据视图、sidecar 审计、统计图、坏样本、run manifest、A4 probe |

## 4. 相较于 PPT 的进阶要求

### A2

| PPT 方向 | 实际完成情况 | 验收时的准确说法 |
|---|---|---|
| LoRA / QLoRA 与资源效果对比 | 已完成 QLoRA，并额外完成 QDoRA；Full 保留为效果参照 | 配置含 rank、alpha、dropout、target、4-bit NF4、double quantization；有显存、速度、参数量和测试结果 |
| LoRA 超参数对比 | 已完成方法维度与 epoch/checkpoint 维度比较，所有超参数均配置化 | 不虚报不存在的完整 rank×alpha 网格；可以说已经建立可复现实验框架，并完成最关键的方法/epoch 对照 |
| 代码错误分析 | 已完成语法、代码抽取、函数签名、exact match/token F1 和执行测试指标；全链路 evaluator 保留逐题失败证据 | 语法正确不等于逻辑正确；以 pass@1 和测试通过率作为核心指标 |
| 代码修复式任务 | 已打通执行反馈和 A5 Reflexion 修复接口 | 当前没有单独训练一个 repair-SFT 模型，因此表述为“完成修复闭环的接口与推理侧验证”，不虚报独立 repair-SFT 结果 |

### A3

| PPT 方向 | 实际完成情况 | 验收时的准确说法 |
|---|---|---|
| 长度统计 | 已完成，而且使用 Qwen3 tokenizer 的 token 长度，并输出分布图 | 比单纯字符平均值更贴近 `cutoff_len` 和真实训练成本 |
| 质量过滤 | 已完成短回答、超长、近重复、语法错误、极端长度差等检查 | 硬无效直接去除，其他风险进入质量分和 flags，避免误删“只差一个关键 bug”的 pair |
| 最短回答控制 | 功能由 `<8 token` 的 `response_too_short` 和 length-safety 分实现 | 当前不是名为 `--min_response_len` 的独立 CLI；不要声称存在这个准确参数名 |
| 95%/5% 切分 | 当前使用 seed 42 shuffle 后固定 500 条 test candidate，实际约为 5% | 为与 default 可比而保留固定数量；报告 prompt overlap。不要说已经实现通用比例 CLI |
| `sample_preview.json` | 已完成 | 同时还有 `bad_pairs.json`、`pair_audit.jsonl` 和 `high_confidence_examples.json` |

## 5. 真实实验结果

### 5.1 A2 正式对照（同一 A2 MBPP 协议，257 题）

| 模型 | pass@1 | Syntax | Avg test pass | 训练耗时 | Peak allocated | 可训练比例 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0/257 = 0% | 94.16% | 0.26% | — | — | 0% |
| Full | 16/257 = 6.23% | 100% | 6.83% | 74.2 min | 9.79 GiB | 100% |
| QLoRA | 9/257 = 3.50% | 99.22% | 4.51% | 89.7 min | 2.58 GiB | 0.8395% |
| QDoRA | 11/257 = 4.28% | 98.44% | 5.41% | 207.9 min | 2.59 GiB | 0.8962% |

证据：

- `sft/outputs/a2_v2_runs/a2_formal_001/reports/comparison_report.md`
- `sft/outputs/a2_v2_runs/a2_formal_001/reports/comparison_report.json`
- `sft/outputs/a2_v2_runs/a2_formal_001/reports/comparison_metrics.png`
- `sft/outputs/a2_v2_runs/a2_formal_001/telemetry/`

解释：Full 是当前效果路线；QLoRA/QDoRA 是资源效率路线。量化 kernel、反量化以及 DoRA magnitude 计算，使 PEFT 在这个 0.6B 小模型和 H200 上不一定更快。

### 5.2 A3 已验证数据漏斗

| 项目 | 数量 |
|---|---:|
| 源数据 | 9,466 |
| Train candidates | 8,966 |
| Test candidates | 500 |
| Raw train | 8,924 |
| Clean train | 8,924 |
| High-confidence train | 8,797 |
| Debug train | 128 |
| Generation test | 485 |

证据：

- `dpo/data/a3_v2_runs/a3_v2_verify_20260702_005837/data_statistics.json`
- 同目录 `preference_quality_report.md`
- 同目录 `pair_audit.jsonl`、`bad_pairs.json`
- 同目录 `reports/filter_funnel.png`、`quality_score_hist.png`、`pair_similarity_hist.png`

### 5.3 完整链路结果必须分协议陈述

旧的隔离 best-path 同协议报告中：A2 Full greedy 为 49.81%，Full→A5 为 65.37%；A4 DPO epoch 2 greedy 为 21.40%，DPO→A5 为 50.58%。证据在：

- `best_path/runs/best_full_dpo_a5_20260702/reports/final_report.md`

组长 7 月 14 日新版 test-time-scaling 协议中：A2 Full greedy 为 46.69%，self-consistency 为 63.04%。证据在：

- `tts/outputs/tts_a2_full_compare/comparison.json`

两组数字不能混成一条提升曲线，因为 prompt 和 evaluator 版本不同。

## 6. 关键实现定位

### A2

- 配置生成、batch、3 epoch、30-step acceptance、QLoRA/QDoRA：`sft/a2_v2/pipeline.py:111-172`
- Qwen3 关闭 thinking：`sft/a2_v2/pipeline.py:123`；正式 YAML 分别在第 10/20/20 行
- 输入校验与 acceptance 抽样：`sft/a2_v2/pipeline.py:65-99`
- RUN_ID 防覆盖：`sft/a2_v2/pipeline.py:178-190`
- checkpoint 选择：`sft/a2_v2/pipeline.py:355-379`
- 代码抽取与 A1 评测：`sft/a2_v2/pipeline.py:397-464`
- telemetry、产物变化验证、报告、handoff：`sft/a2_v2/pipeline.py:477-654`
- 总入口：`sft/scripts/run_a2_v2.sh:37-212`
- bitsandbytes/DoRA 预检：`sft/a2_v2/preflight.py:15-103`
- A2+A3→A4 probe：`sft/scripts/validate_a2_a3_a4_bridge.sh:13-63`

### A3

- Parquet 读取：`dpo/a3_v2/pipeline.py:108-170`
- code extraction / AST：`dpo/a3_v2/pipeline.py:204-235`
- 相似度与质量分：`dpo/a3_v2/pipeline.py:259-357`
- 五种视图：`dpo/a3_v2/pipeline.py:422-538`
- dataset registry：`dpo/a3_v2/pipeline.py:561-603`
- overlap 与质量报告：`dpo/a3_v2/pipeline.py:655-872`
- prepare / validate：`dpo/a3_v2/pipeline.py:886-1116`
- 入口：`dpo/scripts/run_a3_v2.sh:16-78`
- A4 1-step probe：`dpo/scripts/validate_a3_a4_bridge.sh:30-113`

## 7. 总结

我的贡献不只是“跑了两次训练”：A2 把单一 Full SFT 扩展为可复现、可测资源、可交接的多策略模型适配平台；A3 把简单格式转换扩展为可解释、可审计、可分层的偏好数据质量门，并通过稳定文件契约在 A4 汇合。
