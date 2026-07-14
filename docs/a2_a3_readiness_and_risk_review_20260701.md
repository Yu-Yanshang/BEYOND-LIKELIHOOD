# A2 / A3 开工前接口、环境与风险复审报告

> 项目：基于 Qwen3-0.6B 的 Python 代码生成微调与偏好对齐系统  
> 范围：个人 Proposal 中 A2（SFT）与 A3（偏好数据构造），以及 A1 输出、A4 输入接口  
> 复审日期：2026-07-01  
> 项目根目录：`/root/project`  
> Conda 环境：`/opt/conda/envs/shixun`

## 1. 结论摘要

目前可以立即开始 A2/A3 的正式设计与数据治理，但不建议直接启动 Proposal 中的完整对照实验。现状不是“环境完全不可用”，而是基础运行时已通、关键实验契约尚未闭合。

已验证可用的部分：

- Qwen3-0.6B 本地权重、Tokenizer 和 Transformers 接口可读取。
- 单卡 CUDA 张量、Qwen3 bf16 GPU 前向均成功。
- LLaMA-Factory 能读取现有 SFT 数据，完成 1 step 的 LoRA SFT。
- LLaMA-Factory 能读取现有 DPO ranking 数据，完成 1 step 的 LoRA DPO。
- A1 已留下可消费的 Alpaca 格式 train/valid/test JSON。
- A3 当前的最低限度格式转换结果可以被 A4 的 DPO trainer 读取。

开跑正式实验前必须先处理的 P0 问题：

1. **统一 Qwen3 非 thinking 契约。** 对当前混合 thinking/non-thinking 的 Qwen3-0.6B，建议统一使用 `template: qwen3` 加 `enable_thinking: false`，不要继续把 `qwen3_nothink` 当作完全等价替代；自定义推理脚本也必须显式传递 `enable_thinking=False`。
2. **重做 A3 分组切分。** 当前 DPO 数据按“行”随机切分，train/test 有 240 个 prompt 重叠，现有测试集不能作为无泄漏评估集。
3. **闭合 A2→A4 模型接口。** 当前 A4 配置仍从原始 `Qwen3-0.6B` 启动，reference 也是原始基座，没有消费 A2 的 SFT 模型，和 Proposal 的主链路不一致。
4. **恢复量化依赖。** 当前 `shixun` 环境没有 `bitsandbytes`，因此 QLoRA 和 QDoRA 不能直接开跑；普通 LoRA、全量 SFT 不受此项影响。
5. **修复统一评测入口。** SFT 预测配置引用了未注册的 `mbpp_sanitized_test`；评测脚本依赖的 MBPP Parquet 路径也不存在。现在不能形成可信的 Base/SFT/QLoRA/QDoRA 同口径结果。

建议先完成“接口冻结与 A3 治理”，再开始耗时训练。否则训练产物会因模板、泄漏或 reference 错误而需要重做。

## 2. 本次复审方法与证据边界

本次只做环境、接口和最小运行探测，没有实现 A2/A3 正式模块，也没有把历史训练输出作为项目成绩证据。

检查范围包括：

- `docs/a2_a3_proposal.md`
- `sft/configs`、`sft/scripts`、`sft/data`
- `dpo/configs`、`dpo/scripts`、`dpo/data`
- 本地 Qwen3 模型文件和本地 LLaMA-Factory 源码
- A1 源 Parquet、A3 源 Parquet 及当前转换产物
- Qwen3 官方模型卡、PEFT 官方 DoRA 文档、LLaMA-Factory 官方数据契约

本次临时进行了以下无持久化探测：

- Qwen3 bf16 单次 GPU forward；
- 8 条样本、1 step 的 LoRA SFT；
- 8 条样本、1 step 的 LoRA DPO；
- 数据 schema、重复、泄漏、语法代理、长度和 Token 长度统计。

两个训练探测目录均已删除，没有留下临时 checkpoint、脚本或日志。

## 3. 当前运行环境

| 项目 | 当前值 | 判断 |
|---|---:|---|
| GPU | NVIDIA H200 NVL，1 张可见卡 | 可用 |
| PyTorch 可见显存 | 23,552 MiB | 所有配置按单卡约 23 GiB 设计 |
| Driver | 580.119.02 | 当前 CUDA 调用正常 |
| 项目盘剩余 | 约 43 GiB，已使用 84% | 可训练，但必须控制 checkpoint 数量 |
| Python | 3.11.15 | 可用 |
| PyTorch | 2.12.1+cu130 | CUDA 13.0，当前前向/训练探测通过 |
| Transformers | 5.6.0 | 在本地 LLaMA-Factory 声明范围内 |
| LLaMA-Factory | 0.9.6.dev0 | 本地源码可用 |
| PEFT | 0.18.1 | `LoraConfig` 含 `use_dora` |
| Accelerate | 1.11.0 | 可用 |
| Datasets | 4.0.0 | 可用 |
| bitsandbytes | **未安装** | QLoRA/QDoRA 阻断 |

运行探测结果：

- `torch.zeros(1, device="cuda")` 成功，显存查询正常。
- Qwen3 bf16 forward 成功；20 Token 输入时峰值 allocated 约 1.161 GiB、reserved 约 1.184 GiB。
- 1-step LoRA SFT 成功，训练参数 573,440，占总参数约 0.0961%。
- 1-step LoRA DPO 成功，能够输出 `rewards/chosen`、`rewards/rejected`、`rewards/accuracies`、`rewards/margins` 等指标。

这证明模型、GPU、数据解析和基本 trainer 链路已经可用，但不等价于 QLoRA/QDoRA 已可用，也不等价于完整 epoch 配置已经验收。

### 3.1 量化依赖风险

当前环境没有 `bitsandbytes`，`pip check` 不会报告这个问题，因为它是量化路径的可选依赖。正式开始 A2 前应锁定环境快照，并单独验收：

1. `import bitsandbytes` 成功；
2. native CUDA library 确实加载，而不只是 Python import 成功；
3. 4-bit NF4 模型加载成功；
4. 1-step QLoRA 和 1-step QDoRA 均成功；
5. 记录 PyTorch、CUDA、PEFT、bitsandbytes 的确切版本。

此前该机器出现过 CUDA 动态库搜索路径问题，因此恢复 bitsandbytes 后还应检查 CUDA 13 库路径，而不是只看包版本。

## 4. Proposal 功能需求复核

### 4.1 A2 功能就绪度

| Proposal 内容 | 当前状态 | 能否直接正式开动 | 主要缺口 |
|---|---|---|---|
| Qwen3 基座加载、推理、最小训练 | 已验证基础 forward 和 1-step LoRA SFT | 是，作为环境门禁 | 正式推理模板仍需统一 |
| Full SFT baseline | 已有 YAML 和 train 脚本 | 有条件 | thinking 契约、完整评测、资源记录未闭合 |
| QLoRA baseline | 无当前项目正式 config/script | 否 | bitsandbytes 缺失，配置与验收缺失 |
| QDoRA extension | PEFT/LLaMA-Factory 参数接口存在 | 否 | bitsandbytes 缺失，尚无 QDoRA 配置与对照契约 |
| Batch predict | 有 LLaMA-Factory predict 和自定义 infer | 否 | 两条路径的 thinking 行为不一致 |
| MBPP 执行评测 | 有评测代码 | 否 | dataset registry 和 MBPP 路径断裂 |
| Base/Full/QLoRA/QDoRA 公平对比 | 未形成 | 否 | 固定 split、prompt、解码、seed、指标和 run manifest 均待冻结 |
| 产物治理 | 路径命名已有 Qwen3 前缀 | 部分 | 缺少 run manifest、环境快照、数据 hash 和统一 best/final 约定 |

### 4.2 A3 功能就绪度

| Proposal 内容 | 当前状态 | 能否直接正式开动 | 主要缺口 |
|---|---|---|---|
| 读取 py-dpo Parquet | 已实现并验证 | 是 | 应保留原始 `id` 和 provenance |
| 字段标准化 | 已生成 ranking JSON | 是，作为 baseline | 当前转换丢弃了源 `id` |
| 空值/重复清理 | 仅有空字段和整行精确去重 | 不足 | 无 prompt 分组、近重复、代码质量和长度偏差治理 |
| Pair quality audit | 未实现 | 否 | Proposal 中 scorecard 尚无产物 |
| 无泄漏 train/test split | 当前实现不合格 | 否 | 同一 prompt 跨 split |
| raw/clean/high-confidence/debug 多视图 | 未实现 | 否 | 只有一个 train 和一个 generation test |
| data statistics / bad pairs / quality report | 未实现 | 否 | Proposal 约定的 sidecar 文件不存在 |
| A4 trainer 可读性 | 已用 8 条数据、1-step DPO 验证 | 是 | “可读”不代表数据质量合格 |

## 5. A1 输出接口复审

### 5.1 当前接口

当前 A1 风格输出已经存在：

```text
sft/data/code_sft_train.json
sft/data/code_sft_valid.json
sft/data/code_sft_test.json
sft/data/dataset_info.json
sft/data/data_statistics.json
sft/data/sample_preview.json
sft/data/bad_cases.json
```

LLaMA-Factory 字段契约为 Alpaca 格式：

```json
{
  "instruction": "任务描述",
  "input": "可选输入",
  "output": "目标回答"
}
```

`dataset_info.json` 已正确映射 `prompt/query/response`。当前数量为：

- raw：18,612
- 长度过滤后：18,165
- train：16,348
- valid：908
- test：909

### 5.2 数据质量与风险

| 检查 | 结果 | 影响 |
|---|---:|---|
| 原始必需字段空值 | 0 | 基础完整性好 |
| train 内唯一 instruction+input | 16,340 / 16,348 | 有 8 个 prompt/input 重复 |
| train/test prompt+input 重叠 | 1 | 规模小，但说明切分不是按 prompt 分组 |
| raw output 可直接 `ast.parse` | 88.52% | 约 11.5% 不是完整可执行 Python 或存在语法问题 |
| 含函数定义 | 70.35% | 与 MBPP 函数合成并不完全同分布 |
| SFT Token p50 / p99 / max | 111 / 429 / 1,037 | `cutoff_len=2048` 足够，1024 也只影响极少数样本 |

结论：**A1 输出可以作为 A2 的起始输入，但不能无审计地称为“高质量可执行代码数据”。** A2 开工前至少要冻结 A1 数据版本、生成参数、文件 hash，并决定是否沿用现有 split。若最终测试集要承担项目指标，建议按规范化后的 instruction+input 分组重切，避免相同任务跨 split。

不要把 `prompt` 列再次拼进训练输入；当前 JSON 已采用独立的 `instruction/input/output`，这是正确方向。

## 6. A3 原始数据与当前输出复审

### 6.1 原始 py-dpo 数据

原始 Parquet：

- 行数：9,466
- 字段：`prompt`、`chosen`、`rejected`、`id`
- 必需字段空值：0
- 唯一 prompt：6,009
- 唯一 id：6,009
- 最大同 prompt pair 数：7
- 有重复 prompt 的 group：1,487 个，共覆盖 4,944 行

数据集官方卡片明确说明：rejected 是其他模型生成结果，**可能本身也完全正确**，只是总体上假设质量低于 chosen。这意味着 A3 的主要任务不是简单格式转换，而是识别弱偏好、错误偏好和难以区分的 pair。

### 6.2 当前切分的阻断问题

当前脚本先随机打乱 9,466 行，再取前 500 行作为 test，之后才各自去重。最终得到：

- train：8,924
- test：485
- train/test 重叠 prompt：**240**
- test 唯一 prompt：480

即约一半测试 prompt 在训练集中出现过。该 test 只能用于生成接口调试，不能作为 A4 或 A6 的无泄漏评测集。

正确方案是：

1. 保留源 `id`；
2. 以 `id` 为首选 group key，并校验同 id 对应同一规范化 prompt；
3. 若 id 不可信，则使用规范化 prompt hash；
4. 先按 group 切 train/valid/test，再在各 split 内生成多个 pair；
5. 最后强制断言三个 split 的 group key 交集为 0。

### 6.3 偏好质量探测

以下是静态代理，不代表真实功能正确性：

| 指标 | 结果 |
|---|---:|
| chosen 含代码围栏 | 100.00% |
| rejected 含代码围栏 | 82.51% |
| 抽取后 chosen 语法通过 | 97.75% |
| 抽取后 rejected 语法通过 | 92.55% |
| chosen/rejected 都语法通过 | 91.26% |
| 仅 chosen 语法通过 | 6.49% |
| 仅 rejected 语法通过 | 1.29% |
| 两者都不通过 | 0.96% |
| pair 文本/代码相似度 ≥ 0.95 | 2.55% |
| pair 相似度 ≥ 0.80 | 6.94% |
| chosen 比 rejected 更长 | 58.80% |
| chosen-rejected 字符差中位数 | +114 |

这些数据说明原始集有明显偏好信号，但也包含近同质 pair、格式偏差和少量“rejected 语法通过而 chosen 不通过”的可疑样本。语法检查只能作为 flag，不能替代单元测试或人工抽样。

### 6.4 长度与截断

按 Qwen3 ChatML 结构估算当前 DPO train：

| 分支 | p50 | p95 | p99 | 最大 | 超过 1024 | 超过 2048 |
|---|---:|---:|---:|---:|---:|---:|
| chosen | 470 | 1,038 | 1,486 | 3,209 | 5.20% | 0.37% |
| rejected | 435 | 1,487 | 2,093 | 3,062 | 8.51% | 1.46% |

当前 LoRA DPO 配置使用 `cutoff_len: 1024`，会对 chosen/rejected 产生不对称截断，可能把差异最重要的尾部截掉。正式 A4 输入建议默认 2048；超过 2048 的 pair 应被单独标记、过滤或采用明确的保头/保尾策略，而不是静默截断。

### 6.5 推荐的 A3 数据视图

A3 至少应输出以下视图，并全部共享同一 group split：

1. `raw_converted`：只做字段规范化和 provenance 保留；
2. `clean`：去空、去完全重复、去同答案 pair、去跨 split 泄漏；
3. `high_confidence`：排除近同质、明显损坏、chosen 明显弱于 rejected 的 pair；
4. `debug`：从 train group 中固定抽样，供 A4 快速验收；
5. `heldout_generation`：只含未见 prompt，用于模型级生成评估；
6. 可选 `audit_only`：不进入训练但保留用于展示过滤理由。

每条训练记录建议保留或通过 sidecar 关联：

```json
{
  "sample_id": "源 id",
  "group_id": "规范化 prompt hash",
  "instruction": "...",
  "input": "",
  "chosen": "...",
  "rejected": "...",
  "quality_flags": ["chosen_syntax_ok", "near_duplicate"],
  "source": "jondurbin/py-dpo-v0.1"
}
```

LLaMA-Factory 训练 JSON 可只注册所需列；审计字段可留在记录中或放 sidecar，但不能丢失样本到源数据的映射。

## 7. Qwen3 thinking 模式专项结论

### 7.1 官方行为

Qwen3-0.6B 是可在 thinking/non-thinking 间切换的混合模型：

- `enable_thinking=True` 是官方默认值；
- `enable_thinking=False` 是硬关闭；
- 官方说明 thinking 模式不应使用 greedy decoding，推荐 temperature 0.6、top-p 0.95、top-k 20；
- non-thinking 官方推荐 temperature 0.7、top-p 0.8、top-k 20。

当前 SFT 和 DPO 数据均不含真实思维链监督，项目目标又要求较短、可抽取、可执行的 Python 输出。因此 A2/A3/A4 主链路应使用 **non-thinking**。如果 A5 需要 reasoning，应使用原始基座或单独的 reasoning adapter/推理实验，不应让 A2 的代码适配器覆盖唯一基座模型。

### 7.2 本地 LLaMA-Factory 的三种实际编码

本次直接解码了本地 LLaMA-Factory 生成的 prompt/label：

| 配置 | 模型看到的 assistant 前缀 | loss 监督 |
|---|---|---|
| `template: qwen3`, `enable_thinking: true`，样本无 CoT | assistant 后直接开始 response | 自动加入空 `<think></think>`，空块和答案都在 response 侧 |
| `template: qwen3`, `enable_thinking: false` | assistant 后加入空 `<think></think>` | 空块在 prompt 侧被 mask，只对最终答案计算 loss |
| `template: qwen3_nothink` | assistant 后无空 think 块 | 直接对答案计算 loss |

对当前 Qwen3-0.6B，最稳妥的是：

```yaml
template: qwen3
enable_thinking: false
```

理由是它和 Qwen3 官方 `apply_chat_template(..., enable_thinking=False)` 的输入结构一致，并明确把空 think block 放在不计 loss 的 prompt 侧。`qwen3_nothink` 在本地示例中主要搭配 Qwen3-4B-Instruct-2507；该模型是 non-thinking-only 变体，不能直接推导混合版 Qwen3-0.6B 也应使用相同模板。

### 7.3 当前代码中的不一致

现有 YAML 全部写了 `template: qwen3_nothink`，但两个自定义推理路径直接调用：

```python
tokenizer.apply_chat_template(..., add_generation_prompt=True)
```

没有传 `enable_thinking=False`，因此会回到 Qwen3 默认 thinking。MBPP 自定义评测又设置 `do_sample=False`，形成“thinking + greedy”组合，恰好是官方警告的高风险路径，可能导致重复、过长输出、代码抽取失败和结果被低估。

正式实现必须把以下对象视为一个不可拆分的实验契约：

- 训练 template；
- `enable_thinking`；
- 推理 chat template；
- 生成参数；
- code extraction；
- max_new_tokens；
- 评测集和 seed。

不能只改训练 YAML 而不改自定义推理脚本。

## 8. A2 → A4 模型接口

### 8.1 当前问题

当前 DPO 配置使用：

```yaml
model_name_or_path: ./Qwen3-0.6B
ref_model: ./Qwen3-0.6B
```

这绕过 A2。即使 A4 训练成功，也只是“原始 Qwen3 上做 DPO”，不是 Proposal 中“以 A2 适配后的 policy 作为 A4 初始化”。

### 8.2 推荐交付契约

A2 每个正式 run 应至少交付：

```text
run_dir/
  adapter_or_model/
  tokenizer files
  training_args / resolved_config
  trainer_state / trainer_log
  metrics.json
  run_manifest.json
```

`run_manifest.json` 应包含：

- `base_model_path` 和模型文件 hash；
- `method`: full / qlora / qdora；
- `template`、`enable_thinking`；
- 数据集名称、split seed 和数据 hash；
- cutoff、rank、alpha、target modules、quantization 参数；
- best/final checkpoint 路径；
- 是否为完整模型、adapter 或 merged model；
- 环境版本和 git/source snapshot；
- 推荐的 A4 policy 路径和 tokenizer 路径。

### 8.3 推荐的 A4 消费方式

最不容易出错的路径是：

1. A2 的 QLoRA/QDoRA adapter 保留为原始交付物；
2. 在 CPU 上另行导出一个 **A2 merged SFT model**；
3. A4 以 merged SFT model 为 `model_name_or_path`，再创建新的 DPO LoRA adapter；
4. DPO reference 使用“禁用新 DPO adapter 后的 merged SFT model”。

这样 policy 和 reference 都从同一个 A2 SFT 状态出发，且无需同时保留两个完整模型副本。

如果 A4 直接续训 A2 adapter，则必须非常谨慎：本地 LLaMA-Factory 在 LoRA DPO 且未显式提供 reference 时，会通过 `disable_adapter()` 得到 reference；若被禁用的正是包含 SFT 能力的 A2 adapter，reference 会退回原始基座，而不是 SFT policy。可通过 `ref_model_adapters` 显式加载 A2 adapter 作为 reference，但内存、量化和 adapter 合并限制更复杂。故优先推荐 merged SFT 交接。

## 9. 统一评测链路风险

### 9.1 当前断点

- `sft/configs/qwen3_06b_code_full_predict.yaml` 使用 `eval_dataset: mbpp_sanitized_test`，但 `sft/data/dataset_info.json` 没有注册它。
- `sft/scripts/evaluate_full.sh` 默认读取 `/root/project/mbpp`，目录不存在。
- MBPP 转换脚本默认读取 `/root/mbpp`，该目录也不存在。
- `sft/data/mbpp_sanitized_test.json` 虽存在，但没有完整接入 predict/evaluate 契约。
- 自定义 SFT/DPO 推理默认 thinking，且 DPO MBPP 评测使用 greedy。
- 当前 DPO generation test 的 reference 是 chosen 文本，不是单元测试；它适合格式/F1 调试，不足以证明代码功能正确。

### 9.2 建议的单一事实来源

正式实现时应只保留一套固定评测 manifest：

- MBPP 数据文件的确定路径和 hash；
- 固定 task ids；
- 固定 prompt 模板；
- non-thinking 开关；
- max input/output tokens；
- deterministic pass@1 或带固定 seed 的 sampling 协议；
- Base、Full SFT、QLoRA、QDoRA、DPO 完全相同的生成参数；
- `pass_at_1`、syntax、avg test pass、提取成功率、长度、延迟、峰值显存；
- 每个失败样例的类别和原始 completion。

执行模型生成的 Python 属于不可信代码执行。现有 evaluator 使用 subprocess、timeout 和基础资源限制，但仍不等同于安全沙箱。正式批量评测应在无网络、非 root、独立临时目录和受限容器中执行。

## 10. 风险登记与优先级

| 优先级 | 风险 | 概率 | 影响 | 最优处理 |
|---|---|---|---|---|
| P0 | Qwen3 模板/推理 thinking 不一致 | 高 | 训练目标与评测行为错位 | 全链统一 `qwen3 + enable_thinking:false`，加入模板快照测试 |
| P0 | A3 train/test prompt 泄漏 | 已发生 | 指标失真 | 按 id/prompt group 重切并做零交集断言 |
| P0 | A4 未消费 A2 模型 | 已发生 | 主链路不成立 | 冻结 model handoff manifest，优先 merged SFT 交接 |
| P0 | bitsandbytes 缺失 | 已发生 | QLoRA/QDoRA 无法运行 | 锁版本、恢复依赖、做 native CUDA 和 1-step 量化门禁 |
| P0 | MBPP 路径/注册断裂 | 已发生 | 无统一执行指标 | 建立单一 MBPP manifest 和统一 evaluator |
| P1 | DPO cutoff=1024 不对称截断 | 高 | 偏好信号损坏 | 默认 2048，统计并处理超长 pair |
| P1 | rejected 标签本身可能正确 | 数据卡明确说明 | DPO margin 学到噪声 | scorecard、high-confidence 视图、人工/执行抽样 |
| P1 | A1 约 11.5% 输出非直接语法通过 | 中 | SFT 学到不完整代码 | 加质量 flag 和消融，不宜盲目全删 |
| P1 | 历史输出占空间且不可作证据 | 高 | 磁盘和结果污染 | 新 run 使用独立 run_id，不覆盖、不引用历史指标 |
| P1 | 以 root 执行生成代码 | 中 | 安全风险 | 隔离容器、无网络、低权限、资源上限 |
| P2 | 只按 validation loss 选 checkpoint | 高 | 低 loss 但执行差 | 加固定小型执行集，loss 与 pass 指标共同决策 |
| P2 | QDoRA 开销高于 QLoRA | 中 | 时间/显存增加 | 同 rank/step/cutoff 对照，先小预算再扩展 |

## 11. 推荐开工顺序

### Gate 0：环境冻结

- 记录当前版本与 GPU 状态；
- 恢复并验收 bitsandbytes；
- 分别完成 Full/LoRA、QLoRA、QDoRA 1-step 门禁；
- 为磁盘设定 `save_total_limit` 和最小保留空间。

### Gate 1：契约冻结

- 决定主链路 non-thinking；
- 统一 YAML、自定义 infer、MBPP evaluator；
- 定义 A1 数据版本和 A2 run manifest；
- 定义 A2 merged model/adapter 到 A4 的交付方式。

### Gate 2：优先完成 A3

- 保留 id/provenance；
- group split；
- 生成 raw/clean/high-confidence/debug/heldout；
- 输出 scorecard、bad pairs、统计和 split leakage 断言；
- 用 8 条 debug 数据验证 A4 trainer 可读。

### Gate 3：实现 A2 训练梯度

- Base 固定评测；
- Full SFT baseline；
- QLoRA；
- QDoRA；
- 固定预算和相同评测协议，禁止因方法不同改测试 prompt/解码参数。

### Gate 4：A4 联调前验收

- A4 policy 明确指向 A2 产物；
- DPO reference 明确是 A2 初始状态；
- A3 train/valid/test group 零交集；
- 2048 截断统计写入报告；
- 一次 debug DPO 后再交给 A4 跑完整实验。

## 12. 最终就绪度判断

| 状态 | 内容 |
|---|---|
| 绿色：可直接使用 | 本地 Qwen3 权重、CUDA、bf16 forward、普通 LoRA trainer、SFT JSON schema、DPO ranking schema |
| 黄色：修约后可用 | A1 当前 split、Full SFT 配置、当前 DPO 转换 baseline、现有推理和 evaluator 代码 |
| 红色：正式实验前必须补齐 | QLoRA/QDoRA 环境、A3 group split/scorecard、多视图数据、A2→A4 model/reference 接口、统一 MBPP 评测、全链 thinking 契约 |

总体判断：**可以开始 A2/A3 实现工作，但第一批提交应是环境门禁、接口契约和 A3 数据治理，而不是直接跑完整 epoch。** 当前硬件足以支撑 Qwen3-0.6B；最大风险来自模板、数据泄漏、reference 定义和评测一致性，而不是模型尺寸本身。

## 13. 参考资料

1. Qwen Team, Qwen3-0.6B Model Card：<https://huggingface.co/Qwen/Qwen3-0.6B>
2. Qwen3 官方仓库：<https://github.com/QwenLM/Qwen3>
3. LLaMA-Factory 官方仓库：<https://github.com/hiyouga/LLaMA-Factory>
4. LLaMA-Factory 数据格式说明：<https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md>
5. Hugging Face PEFT LoRA/DoRA 文档：<https://huggingface.co/docs/peft/main/package_reference/lora>
6. py-dpo-v0.1 数据卡：<https://huggingface.co/datasets/jondurbin/py-dpo-v0.1>
7. Direct Preference Optimization：<https://arxiv.org/abs/2305.18290>

