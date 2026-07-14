# PathCoder

PathCoder 是本项目的 Python 代码推演与验证工作台。它把 Qwen3-0.6B Full SFT
模型、七种 Test-Time Scaling 策略、受限执行验证、指标目录和会话轨迹放进一个
统一入口。

## 快速开始

登录服务器后无需切换目录或激活 Conda：

```bash
pathcoder
```

首次使用建议先检查环境，再运行快速示例：

```bash
pathcoder doctor
pathcoder run --example two_sum --strategy greedy
```

质量优先的综合演示：

```bash
pathcoder run --example merge_intervals --strategy hybrid
```

## 命令导航

```text
pathcoder                         启动交互工作台
pathcoder run [选项]              运行单次任务
pathcoder examples                查看内置任务
pathcoder strategies              比较七种推理策略
pathcoder history                 查看最近会话
pathcoder doctor                  检查模型、GPU 和输出目录
pathcoder performance             查看 best_path 正式成绩
pathcoder metrics [--all]         查看项目指标
pathcoder analyze                 查看效果和原因分析
pathcoder export [路径]           导出规范化指标 JSON
pathcoder version                 显示版本
```

`final` 和 `final-system` 作为兼容入口继续可用。新文档和展示统一使用
`pathcoder`。

## 自定义任务

```bash
pathcoder run \
  --strategy hybrid \
  --task "Write a function factorial(n) that raises ValueError for n < 0." \
  --test "assert factorial(0) == 1" \
  --test "assert factorial(5) == 120"
```

每个 `--test` 必须是 Python `assert`。未提供测试时，系统只能按语法、格式和
候选一致性进行选择，不能宣称功能正确。

## 推理策略

- `greedy`: 单次确定性生成，速度最快。
- `cot`: 先组织关键步骤，再生成代码。
- `self_consistency`: 多路径采样和代码一致性投票。
- `best_of_n`: 多候选生成后按语法和测试重排。
- `reflexion`: 根据执行失败反馈修正代码。
- `tot`: 搜索并评分多条解题思路。
- `hybrid`: ToT、Self-Consistency、Best-of-N 和 Reflexion 的综合流程。

`hybrid` 质量优先且耗时更长。现场快速验证优先使用 `greedy`，需要完整展示时
再使用 `hybrid`。

## 输出与可复现性

会话默认写入：

```text
/root/project/tts/outputs/showcase_sessions/
```

每条 JSON 包含任务、测试、策略、候选、执行反馈、最终代码、耗时和 token 使用。
使用 `pathcoder history --history_limit 20` 查看最近记录。

正式成绩读取自：

```text
/root/project/best_path/runs/best_full_dpo_a5_20260702/reports/final_report.json
```

该报告中 A2 Full greedy 的 MBPP Sanitized pass@1 为 49.81%，同一模型经 A5
增强后为 65.37%。现场单题结果不等同于全量 benchmark。

## 高级参数

```bash
pathcoder run --help
```

常用成本参数包括 `--best_of_n`、`--tree_width`、`--tree_depth`、
`--reflexion_rounds`、`--max_new_tokens` 和 `--test_timeout`。
