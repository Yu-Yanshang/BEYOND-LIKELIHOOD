# Qwen3-0.6B Python Code DPO

本目录是在 `/root/project` 工作区内完成的大作业基础版本：使用 LLaMA-Factory 对本地 `Qwen3-0.6B` 进行 Python 代码任务偏好优化。整体流程参考 `shixun-dpo`，但任务从数学推理改为代码生成。

所有配置都使用 `/root/project` 内的相对路径，例如 `./Qwen3-0.6B`、`./py-dpo-v0.1`、`./dpo/data`、`./dpo/outputs/...`。

## 数据

原始数据：`./py-dpo-v0.1/py-dpo.parquet`

原始字段：

- `prompt`：代码任务描述
- `chosen`：更优的代码回答
- `rejected`：较差的代码回答
- `id`：样本 id

转换后会生成 LLaMA-Factory ranking 数据：

```json
{
  "instruction": "Complete the following Python coding task. ...",
  "input": "",
  "chosen": "better Python solution",
  "rejected": "worse Python solution"
}
```

同时会随机抽取一部分样本作为生成测试集：

```json
{
  "instruction": "Complete the following Python coding task. ...",
  "input": "",
  "output": "reference Python solution"
}
```

## 目录

- `configs/qwen3_06b_code_full_dpo.yaml`：代码任务 DPO 训练配置
- `configs/qwen3_06b_code_predict_base.yaml`：DPO 前 base 模型生成配置
- `configs/qwen3_06b_code_predict_dpo.yaml`：DPO 后模型生成配置
- `scripts/prepare_dpo_data.py`：将 parquet 转为 LLaMA-Factory DPO 数据
- `scripts/prepare_data.sh`：数据准备入口
- `scripts/train.sh`：DPO 训练入口
- `scripts/test_code_dpo.py`：唯一测试脚本，生成并比较 base/DPO 代码输出
- `scripts/run_all.sh`：一键执行数据准备、训练和测试

## 运行

```bash
cd /root/project

bash dpo/scripts/prepare_data.sh
bash dpo/scripts/train.sh
python3 dpo/scripts/test_code_dpo.py
```

一键运行：

```bash
bash dpo/scripts/run_all.sh
```

## 小样本调试

```bash
MAX_TRAIN_SAMPLES=200 TEST_SIZE=50 bash dpo/scripts/prepare_data.sh
bash dpo/scripts/train.sh
python3 dpo/scripts/test_code_dpo.py
```

指定 GPU：

```bash
GPU_ID=0 bash dpo/scripts/train.sh
python3 dpo/scripts/test_code_dpo.py --gpu_id 0
```

如果已经生成过预测，只想重新计算指标：

```bash
python3 dpo/scripts/test_code_dpo.py --skip_predict
```

## 测试指标

测试脚本不会执行模型生成的代码，只做静态检查和文本级比较：

- `syntax_pass_rate`：生成代码能否通过 `ast.parse`
- `exact_match`：提取代码后与参考答案是否完全一致
- `avg_code_token_f1`：代码 token 级别 F1

## 输出

- 训练数据：`dpo/data/code_dpo_train.json`
- 测试数据：`dpo/data/code_dpo_test.json`
- 数据注册：`dpo/data/dataset_info.json`
- DPO 模型：`dpo/outputs/qwen3_06b_code_full_dpo`
- base 预测：`dpo/outputs/predict_qwen3_base_on_code_dpo/generated_predictions.jsonl`
- DPO 预测：`dpo/outputs/predict_qwen3_dpo_on_code_dpo/generated_predictions.jsonl`
- 测试汇总：`dpo/outputs/code_test/code_dpo_test_metrics.json`
- 前后对比样例：`dpo/outputs/code_test/code_dpo_compare_cases.jsonl`
