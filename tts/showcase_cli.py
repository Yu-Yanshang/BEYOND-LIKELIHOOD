#!/usr/bin/env python3
"""Interactive project showcase for the A2 Full Python code model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

try:
    from evaluation_catalog import (
        MetricRecord,
        discover_metric_records,
        export_catalog,
        full_evaluations,
        merged_training_runs,
        running_evaluation_jobs,
    )
except ImportError:
    from tts.evaluation_catalog import (
        MetricRecord,
        discover_metric_records,
        export_catalog,
        full_evaluations,
        merged_training_runs,
        running_evaluation_jobs,
    )

try:
    from test_time_scaling import (
        ModelGenerator,
        append_instruction,
        canonical_code,
        compact_candidate,
        evaluate_candidate,
        feedback_from_case,
        final_case,
        run_strategy,
        save_json,
        score_thought,
        select_best,
        summarize,
        thought_messages,
    )
except ImportError:
    from tts.test_time_scaling import (
        ModelGenerator,
        append_instruction,
        canonical_code,
        compact_candidate,
        evaluate_candidate,
        feedback_from_case,
        final_case,
        run_strategy,
        save_json,
        score_thought,
        select_best,
        summarize,
        thought_messages,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_RUN_ROOT = PROJECT_ROOT / "best_path" / "runs" / "best_full_dpo_a5_20260702"
BEST_REPORT = BEST_RUN_ROOT / "reports" / "final_report.json"
DEFAULT_MODEL = BEST_RUN_ROOT / "models" / "a2_full"
DEFAULT_OUTPUT = PROJECT_ROOT / "tts" / "outputs" / "showcase_sessions"
STRATEGIES = ("greedy", "cot", "self_consistency", "best_of_n", "reflexion", "tot", "hybrid")
APP_NAME = "PathCoder"
APP_VERSION = "1.0.0"
APP_TAGLINE = "Python 代码推演与验证工作台"

EXAMPLES: dict[str, dict[str, Any]] = {
    "two_sum": {
        "title": "Two Sum - 哈希表查找",
        "task": (
            "Write a function two_sum(nums, target) that returns the indices of two distinct "
            "numbers whose sum equals target. Return an empty list if no pair exists."
        ),
        "tests": [
            "assert two_sum([2, 7, 11, 15], 9) == [0, 1]",
            "assert two_sum([3, 2, 4], 6) == [1, 2]",
            "assert two_sum([1, 2, 3], 100) == []",
        ],
        "feature": "展示函数签名遵循、哈希表算法和边界处理",
    },
    "palindrome": {
        "title": "Palindrome - 字符串清洗",
        "task": (
            "Write a function is_palindrome(text) that checks whether text is a palindrome, "
            "ignoring case and all non-alphanumeric characters."
        ),
        "tests": [
            "assert is_palindrome('A man, a plan, a canal: Panama') is True",
            "assert is_palindrome('race a car') is False",
            "assert is_palindrome('') is True",
        ],
        "feature": "展示文本处理、正则/字符判断和空输入边界",
    },
    "merge_intervals": {
        "title": "Merge Intervals - 排序与区间合并",
        "task": (
            "Write a function merge_intervals(intervals) that merges every overlapping closed "
            "interval and returns the merged intervals sorted by start value."
        ),
        "tests": [
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]",
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]",
            "assert merge_intervals([]) == []",
        ],
        "feature": "展示较复杂控制流、排序、闭区间语义和空列表处理",
    },
    "frequency": {
        "title": "Word Frequency - 字典统计",
        "task": (
            "Write a function word_frequency(words) that returns a dictionary mapping each word "
            "to its occurrence count. Preserve the original spelling and handle an empty list."
        ),
        "tests": [
            "assert word_frequency(['a','b','a']) == {'a': 2, 'b': 1}",
            "assert word_frequency([]) == {}",
            "assert word_frequency(['A','a']) == {'A': 1, 'a': 1}",
        ],
        "feature": "展示基础 Python 代码质量、字典操作和大小写语义",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pathcoder",
        description=f"{APP_NAME} - {APP_TAGLINE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "常用示例:\n"
            "  pathcoder                         启动交互工作台\n"
            "  pathcoder run --example two_sum   运行内置任务\n"
            "  pathcoder doctor                  检查运行环境\n"
            "  pathcoder history                 查看最近会话\n"
            "更多子命令请运行 pathcoder help。"
        ),
    )
    general = parser.add_argument_group("任务与界面")
    general.add_argument("--strategy", choices=STRATEGIES, default="hybrid", help="推理策略，默认 hybrid。")
    general.add_argument("--example", choices=tuple(EXAMPLES), help="运行指定内置任务。")
    general.add_argument("--task", help="单次任务描述；不提供时进入交互模式。")
    general.add_argument("--test", action="append", default=[], help="Python assert，可重复提供。")
    general.add_argument("--list_examples", action="store_true", help="列出内置任务。")
    general.add_argument("--show_strategies", action="store_true", help="显示推理策略。")
    general.add_argument("--show_performance", action="store_true", help="显示核心性能。")
    general.add_argument("--show_all_metrics", action="store_true", help="显示项目指标目录。")
    general.add_argument("--analyze_results", action="store_true", help="分析训练与推理效果。")
    general.add_argument("--show_history", action="store_true", help="显示最近会话。")
    general.add_argument("--history_limit", type=int, default=8, help="历史会话显示条数，默认 8。")
    general.add_argument("--doctor", action="store_true", help="检查模型、GPU 和输出目录。")
    general.add_argument("--include_smoke", action="store_true", help="指标中包含 acceptance/probe/smoke。")
    general.add_argument("--export_catalog", type=Path, help="将规范化指标导出为 JSON。")
    runtime = parser.add_argument_group("高级运行参数")
    runtime.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    runtime.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    runtime.add_argument("--batch_size", type=int, default=4)
    runtime.add_argument("--max_input_tokens", type=int, default=3072)
    runtime.add_argument("--max_new_tokens", type=int, default=512)
    runtime.add_argument("--temperature", type=float, default=0.8)
    runtime.add_argument("--top_p", type=float, default=0.95)
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device_map", default="auto")
    runtime.add_argument("--num_samples", type=int, default=3)
    runtime.add_argument("--best_of_n", type=int, default=4)
    runtime.add_argument("--reflexion_rounds", type=int, default=2)
    runtime.add_argument("--tree_width", type=int, default=2)
    runtime.add_argument("--tree_depth", type=int, default=1)
    runtime.add_argument("--test_timeout", type=float, default=5.0)
    runtime.add_argument("--memory_mb", type=int, default=1024)
    runtime.add_argument("--no_color", action="store_true")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    args = parser.parse_args()
    args.input_file = Path("interactive-input")
    for name in ("batch_size", "max_input_tokens", "max_new_tokens", "num_samples", "best_of_n", "tree_width", "tree_depth", "memory_mb", "history_limit"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.reflexion_rounds < 0:
        parser.error("--reflexion_rounds must be non-negative")
    invalid_tests = [test for test in args.test if not test.lstrip().startswith("assert ")]
    if invalid_tests:
        parser.error("--test must contain a Python assert statement")
    return args


def make_console(args: argparse.Namespace) -> Console:
    return Console(no_color=args.no_color, highlight=False)


def banner(console: Console) -> None:
    title = Text(APP_NAME.upper(), style="bold bright_cyan")
    subtitle = Text(APP_TAGLINE, style="bright_white")
    stack = Text("Qwen3-0.6B  |  Full SFT  |  Test-Time Scaling", style="dim")
    console.print(Panel(Text.assemble(title, "\n", subtitle, "\n", stack), border_style="cyan", padding=(1, 4)))


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def best_path_report() -> dict[str, Any]:
    """Load the immutable best_path selection/evaluation report."""
    return load_json(BEST_REPORT)


def performance_rows() -> list[dict[str, Any]]:
    metrics = best_path_report().get("metrics", {})
    candidates = [
        ("A2 Full · greedy", metrics.get("a2_full_greedy")),
        ("A2 Full · A5 enhanced", metrics.get("a5_from_a2_full")),
    ]
    return [{"label": label, **item} for label, item in candidates if item]


def show_performance(console: Console, args: argparse.Namespace) -> None:
    model_dir = args.model_path
    report = best_path_report()
    manifest = report.get("manifest", {})
    config = load_json(model_dir / "config.json")
    train = load_json(model_dir / "train_results.json")
    evaluation = load_json(model_dir / "eval_results.json")

    facts = Table(title="模型档案", box=box.ROUNDED, border_style="cyan", show_header=False)
    facts.add_column(style="bold cyan")
    facts.add_column()
    facts.add_row("模型路径", str(model_dir))
    facts.add_row("Best-path run", str(report.get("run_id", "best_full_dpo_a5_20260702")))
    facts.add_row("架构", str((config.get("architectures") or ["Qwen3ForCausalLM"])[0]))
    facts.add_row("训练方式", "A2 Full fine-tuning · 2 epochs")
    facts.add_row("训练数据", f"A1 Python 指令 {manifest.get('sources', {}).get('a1_train.json', {}).get('rows', 'n/a')} 条")
    facts.add_row("训练损失", f"{train.get('train_loss', 'n/a')}")
    facts.add_row("验证损失", f"{evaluation.get('eval_loss', 'n/a')}")
    facts.add_row("核心能力", "Python 代码生成、CoT、执行验证、反思修正、搜索式推理")
    console.print(facts)

    rows = performance_rows()
    if rows:
        table = Table(title="best_path 正式 MBPP Sanitized 评测（257 题）", box=box.ROUNDED, border_style="green")
        table.add_column("模型", style="bold")
        table.add_column("Syntax", justify="right")
        table.add_column("pass@1", justify="right")
        table.add_column("通过题数", justify="right")
        table.add_column("平均测试通过", justify="right")
        for row in rows:
            table.add_row(
                row["label"],
                f"{100 * row['syntax_pass_rate']:.2f}%",
                f"{100 * row['pass_at_1']:.2f}%",
                f"{row['passed_tasks']}/{row['total']}",
                f"{100 * row['avg_test_pass_rate']:.2f}%",
            )
        console.print(table)
        if len(rows) >= 2 and rows[0].get("pass_at_1"):
            absolute = rows[1]["pass_at_1"] - rows[0]["pass_at_1"]
            relative = absolute / rows[0]["pass_at_1"]
            console.print(
                f"[green]同一 A2 Full 模型加入 A5 推理增强后：pass@1 绝对提升 {absolute * 100:.2f} 个百分点，"
                f"相对提升 {relative * 100:.2f}%[/green]"
            )
    contract = manifest.get("contract", {}).get("a5")
    if contract:
        console.print(Panel(contract, title="best_path A5 正式策略", border_style="green"))
    console.print(f"[dim]数据源：{BEST_REPORT}[/dim]")
    console.print("[dim]注：上表来自 best_path 保存的统一协议全量结果；现场单题结果不等同于全量 benchmark。[/dim]")


def show_home(console: Console, args: argparse.Namespace) -> None:
    """Show a compact start screen instead of the full benchmark report."""
    rows = performance_rows()
    enhanced = rows[-1] if rows else {}
    model_ready = (args.model_path / "config.json").is_file()
    recent_count = len(list(args.output_dir.glob("*.json"))) if args.output_dir.is_dir() else 0
    status = "[green]就绪[/green]" if model_ready else "[red]模型缺失[/red]"
    score = "n/a"
    if enhanced.get("pass_at_1") is not None:
        score = f"{100 * enhanced['pass_at_1']:.2f}%"
    body = (
        f"状态  {status}    核心评测 pass@1  [bold]{score}[/bold]    已保存会话  [bold]{recent_count}[/bold]\n\n"
        "[bold]直接开始[/bold]  选择下方 1 或 2\n"
        "[dim]命令提示：pathcoder run --example merge_intervals --strategy hybrid[/dim]"
    )
    console.print(Panel(body, title="工作台", border_style="cyan"))


def recent_sessions(output_dir: Path, limit: int = 8) -> list[dict[str, Any]]:
    """Return lightweight summaries for recent, readable session files."""
    if not output_dir.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = load_json(path)
        if not payload:
            continue
        result = payload.get("result") or {}
        metrics = payload.get("metrics") or {}
        sessions.append(
            {
                "path": path,
                "created_at": str(payload.get("created_at") or ""),
                "strategy": str(payload.get("strategy") or "unknown"),
                "task": str(payload.get("task") or "").strip(),
                "syntax_ok": bool(result.get("syntax_ok")),
                "passed": bool(result.get("passed")),
                "total_tests": int(result.get("total_tests") or 0),
                "passed_tests": int(result.get("passed_tests") or 0),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
            }
        )
        if len(sessions) >= limit:
            break
    return sessions


def show_history(console: Console, args: argparse.Namespace) -> None:
    sessions = recent_sessions(args.output_dir, args.history_limit)
    if not sessions:
        console.print(
            Panel(
                "还没有可显示的会话。运行 [bold]pathcoder run --example two_sum[/bold] 创建第一条记录。",
                title="最近会话",
                border_style="yellow",
            )
        )
        return
    table = Table(title=f"最近会话（{len(sessions)} 条）", box=box.ROUNDED, border_style="cyan")
    table.add_column("时间", style="dim", no_wrap=True)
    table.add_column("策略", style="bold cyan")
    table.add_column("结果", justify="center")
    table.add_column("耗时", justify="right")
    table.add_column("任务", overflow="fold")
    for item in sessions:
        created = item["created_at"].replace("T", " ")[:19] or item["path"].stem[:15]
        if item["total_tests"]:
            result = f"{item['passed_tests']}/{item['total_tests']}"
        else:
            result = "语法通过" if item["syntax_ok"] else "语法失败"
        elapsed = item["elapsed_seconds"]
        table.add_row(
            created,
            item["strategy"],
            result,
            "-" if elapsed is None else f"{float(elapsed):.1f}s",
            re.sub(r"\s+", " ", item["task"])[:72],
        )
    console.print(table)
    console.print(f"[dim]会话目录：{args.output_dir}[/dim]")


def doctor_checks(args: argparse.Namespace) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("项目目录", PROJECT_ROOT.is_dir(), str(PROJECT_ROOT)))
    checks.append(("模型配置", (args.model_path / "config.json").is_file(), str(args.model_path)))
    checks.append(("正式报告", BEST_REPORT.is_file(), str(BEST_REPORT)))
    output_parent = args.output_dir if args.output_dir.exists() else args.output_dir.parent
    checks.append(("会话输出", output_parent.is_dir() and os.access(output_parent, os.W_OK), str(args.output_dir)))
    checks.append(("Rich 终端", importlib.util.find_spec("rich") is not None, "rich"))
    try:
        import torch

        cuda_ready = torch.cuda.is_available()
        if cuda_ready:
            detail = f"{torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | torch {torch.__version__}"
        else:
            detail = f"未检测到 CUDA | torch {torch.__version__}"
        checks.append(("GPU", cuda_ready, detail))
    except Exception as exc:
        checks.append(("GPU", False, f"PyTorch 检查失败：{exc}"))
    try:
        free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
        checks.append(("磁盘空间", free_gb >= 2.0, f"可用 {free_gb:.1f} GiB"))
    except OSError as exc:
        checks.append(("磁盘空间", False, str(exc)))
    return checks


def show_doctor(console: Console, args: argparse.Namespace) -> bool:
    checks = doctor_checks(args)
    table = Table(title="PathCoder 环境检查", box=box.ROUNDED, border_style="cyan")
    table.add_column("状态", justify="center")
    table.add_column("检查项", style="bold")
    table.add_column("详情", overflow="fold")
    for label, ok, detail in checks:
        table.add_row("[green]通过[/green]" if ok else "[red]失败[/red]", label, detail)
    console.print(table)
    healthy = all(ok for _, ok, _ in checks)
    if healthy:
        console.print("[bold green]环境可用，可以开始运行任务。[/bold green]")
    else:
        console.print("[bold red]环境尚未就绪，请先处理上表中的失败项。[/bold red]")
    return healthy


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def compact_label(label: str, limit: int = 54) -> str:
    return label if len(label) <= limit else "…" + label[-(limit - 1) :]


def show_metric_table(console: Console, records: list[MetricRecord], title: str) -> None:
    table = Table(title=title, box=box.ROUNDED, border_style="cyan", show_lines=False)
    table.add_column("来源", style="bold", overflow="fold")
    table.add_column("训练", overflow="fold")
    table.add_column("策略")
    table.add_column("协议", overflow="fold")
    table.add_column("N", justify="right")
    table.add_column("pass@1", justify="right")
    table.add_column("Syntax", justify="right")
    table.add_column("Tests", justify="right")
    table.add_column("s/task", justify="right")
    ordered = sorted(
        records,
        key=lambda item: (
            {"best_path": 0, "sft": 1, "dpo": 2, "tts": 3}.get(item.module, 9),
            item.protocol,
            -(item.pass_at_1 if item.pass_at_1 is not None else -1),
            item.label,
        ),
    )
    for record in ordered:
        table.add_row(
            compact_label(record.label, 42),
            record.training_method,
            record.strategy,
            record.protocol,
            str(record.tasks or "-"),
            percent(record.pass_at_1),
            percent(record.syntax_pass_rate),
            percent(record.avg_test_pass_rate),
            "-" if record.latency_seconds is None else f"{record.latency_seconds:.2f}",
        )
    console.print(table)


def show_training_table(console: Console, records: list[MetricRecord]) -> None:
    runs = merged_training_runs(records)
    table = Table(title=f"训练记录（{len(runs)} 个模型/运行目录）", box=box.ROUNDED, border_style="magenta")
    table.add_column("模块", style="bold")
    table.add_column("运行目录", overflow="fold")
    table.add_column("方法")
    table.add_column("Epoch", justify="right")
    table.add_column("Train loss", justify="right")
    table.add_column("Eval loss", justify="right")
    table.add_column("训练秒数", justify="right")
    for item in runs:
        table.add_row(
            item["module"],
            compact_label(item["label"], 58),
            item["training_method"],
            "-" if item["epoch"] is None else f"{item['epoch']:.1f}",
            "-" if item["train_loss"] is None else f"{item['train_loss']:.4f}",
            "-" if item["eval_loss"] is None else f"{item['eval_loss']:.4f}",
            "-" if item["train_runtime"] is None else f"{item['train_runtime']:.1f}",
        )
    console.print(table)
    console.print("[dim]不同训练目标的 loss 定义不同，SFT、DPO、PPO 的 loss 数值不能直接横向排名。[/dim]")


def show_all_metrics(console: Console, args: argparse.Namespace) -> list[MetricRecord]:
    records = discover_metric_records(PROJECT_ROOT)
    evaluations = full_evaluations(records, include_smoke=args.include_smoke)
    full_count = sum(record.kind == "evaluation" and not record.is_smoke for record in records)
    smoke_count = sum(record.kind == "evaluation" and record.is_smoke for record in records)
    console.print(
        Panel(
            f"扫描到 [bold]{len(records)}[/bold] 条可解析指标：正式评估 {full_count} 条，"
            f"probe/acceptance/smoke {smoke_count} 条。\n"
            f"当前显示：{'全部（含小样本）' if args.include_smoke else '正式/全量记录'}。",
            title="全项目指标目录",
            border_style="cyan",
        )
    )
    for module in ("best_path", "sft", "dpo", "tts"):
        subset = [record for record in evaluations if record.module == module]
        if subset:
            show_metric_table(console, subset, f"{module} · {len(subset)} 条评估")
    other = [record for record in evaluations if record.module not in ("best_path", "sft", "dpo", "tts")]
    if other:
        show_metric_table(console, other, f"其他模块 · {len(other)} 条评估")
    show_training_table(console, records)
    jobs = running_evaluation_jobs()
    if jobs:
        console.print(Panel("\n".join(jobs), title="仍在运行的评估（未进入最终排名）", border_style="yellow"))
    return records


def find_record(records: list[MetricRecord], path_fragment: str) -> MetricRecord | None:
    return next((record for record in records if path_fragment in record.relative_path), None)


def show_effect_analysis(console: Console, args: argparse.Namespace) -> None:
    records = discover_metric_records(PROJECT_ROOT)
    full = full_evaluations(records, include_smoke=False)
    console.rule("[bold cyan]1. 监督微调：Base / Full / QLoRA / QDoRA")
    sft_formal = [
        record
        for record in full
        if "sft/outputs/a2_v2_runs/a2_formal_001/eval/final/" in record.relative_path
        and record.relative_path.endswith("mbpp_metrics.json")
    ]
    show_metric_table(console, sft_formal, "同一 LLaMA-Factory prediction 协议")
    sft_by_method = {record.training_method: record for record in sft_formal}
    full_sft = sft_by_method.get("Full SFT")
    base_sft = sft_by_method.get("Base")
    if full_sft and base_sft:
        console.print(
            Panel(
                f"Full SFT 将 pass@1 从 {percent(base_sft.pass_at_1)} 提升到 {percent(full_sft.pass_at_1)}，"
                f"并把函数签名/代码格式学习进模型。QLoRA/QDoRA 参数容量更小，当前质量低于 Full，"
                "但训练和保存成本更低。",
                title="原因解析",
                border_style="green",
            )
        )

    console.rule("[bold cyan]2. 偏好与强化学习：DPO / LoRA DPO / PPO")
    dpo_full = [record for record in full if record.module == "dpo" and record.pass_at_1 is not None]
    show_metric_table(console, dpo_full, "DPO/PPO 全量记录（协议列必须同时查看）")
    if dpo_full:
        best_dpo = max(dpo_full, key=lambda record: record.pass_at_1 or 0.0)
        console.print(
            Panel(
                f"当前 dpo 目录最高记录是 {best_dpo.label}，pass@1={percent(best_dpo.pass_at_1)}。"
                "偏好/RL 训练并非必然提升代码执行率：奖励若更偏格式、长度或局部测试，可能损伤函数签名、"
                "语法和泛化；旧 DPO 记录还使用 Google MBPP legacy prompt，不能与 best_path 数值直接比较。",
                title="原因解析",
                border_style="yellow",
            )
        )

    console.rule("[bold cyan]3. 最佳训练链与推理增强")
    best_rows = [
        record
        for record in full
        if record.relative_path.endswith("best_path/runs/best_full_dpo_a5_20260702/eval/a2_full/metrics.json")
        or record.relative_path.endswith("best_path/runs/best_full_dpo_a5_20260702/eval/a5_from_a2_full/metrics.json")
        or record.relative_path.endswith("best_path/runs/best_full_dpo_a5_20260702/eval/a4_dpo/metrics.json")
        or record.relative_path.endswith("best_path/runs/best_full_dpo_a5_20260702/eval/a5_from_a4_dpo/metrics.json")
    ]
    show_metric_table(console, best_rows, "best_path 正式统一协议")
    a2 = find_record(records, "best_path/runs/best_full_dpo_a5_20260702/eval/a2_full/metrics.json")
    a5 = find_record(records, "best_path/runs/best_full_dpo_a5_20260702/eval/a5_from_a2_full/metrics.json")
    if a2 and a5 and a2.pass_at_1 is not None and a5.pass_at_1 is not None:
        gain = a5.pass_at_1 - a2.pass_at_1
        console.print(
            Panel(
                f"A2 Full 从 greedy {percent(a2.pass_at_1)} 提升到 A5 enhanced {percent(a5.pass_at_1)}，"
                f"绝对提升 {100 * gain:.2f} 个百分点。主要来源是多样化候选扩大正确解覆盖率，"
                "执行 verifier 消除语法/测试失败，再由 Reflexion 修复残余错误。",
                title="最佳路线效果",
                border_style="green",
            )
        )

    console.rule("[bold cyan]4. Test-Time Scaling 成本与收益")
    tts_full = [record for record in full if record.module == "tts" and "tts/outputs/tts_" in record.relative_path]
    show_metric_table(console, tts_full, "A2 Full · TTS v2 全量结果")
    greedy = find_record(records, "tts/outputs/tts_a2_full_compare/greedy/metrics.json")
    cot = find_record(records, "tts/outputs/tts_a2_full_compare/cot/metrics.json")
    sc = find_record(records, "tts/outputs/tts_self_consistency/self_consistency/metrics.json")
    bon = find_record(records, "tts/outputs/tts_best_of_n/best_of_n/metrics.json")
    reflex = find_record(records, "tts/outputs/tts_reflexion/reflexion/metrics.json")
    observations = []
    if greedy and sc:
        observations.append(
            f"Self-Consistency：{percent(greedy.pass_at_1)} → {percent(sc.pass_at_1)}，"
            f"但延迟约为 {sc.latency_seconds / greedy.latency_seconds:.1f}×。"
        )
    if greedy and bon:
        observations.append(
            f"Best-of-N：{percent(bon.pass_at_1)}，比 Self-Consistency 略低但更快，且 verifier 选择更稳定。"
        )
    if greedy and cot:
        observations.append(
            f"单独 CoT 降至 {percent(cot.pass_at_1)}：严格分段输出增加格式/提取错误，不代表模型能力下降。"
        )
    if cot and reflex:
        observations.append(
            f"Reflexion 将 CoT 的 {percent(cot.pass_at_1)} 修复至 {percent(reflex.pass_at_1)}，"
            "适合只处理多候选仍失败的题目。"
        )
    console.print(Panel("\n".join(f"• {line}" for line in observations), title="效果与原因", border_style="cyan"))

    console.rule("[bold red]5. 评测解释边界")
    console.print(
        Panel(
            "• 只在相同 protocol、数据集和 prompt 下比较训练方法。\n"
            "• visible-tests 协议把测试放入 prompt 并用于重排，指标应理解为测试感知选择成功率。\n"
            "• acceptance/probe/smoke 默认不进入排行榜。\n"
            "• SFT、DPO、PPO 的 loss 含义不同，不能用 loss 大小直接判断模型优劣。\n"
            "• 当前 Tree of Thoughts 若仍在运行，会显示在任务状态中，完成前不参与结论。",
            border_style="red",
        )
    )
    jobs = running_evaluation_jobs()
    if jobs:
        console.print(Panel("\n".join(jobs), title="运行中", border_style="yellow"))


def show_examples(console: Console) -> None:
    table = Table(title="内置演示任务", box=box.ROUNDED, border_style="magenta")
    table.add_column("编号", justify="right", style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("任务")
    table.add_column("展示重点")
    for number, (key, item) in enumerate(EXAMPLES.items(), 1):
        table.add_row(str(number), key, item["title"], item["feature"])
    console.print(table)


def show_strategies(console: Console) -> None:
    table = Table(title="推理策略", box=box.ROUNDED, border_style="yellow")
    table.add_column("策略", style="bold cyan")
    table.add_column("机制")
    table.add_column("适合场景")
    descriptions = [
        ("greedy", "单次确定性生成", "最快速现场响应"),
        ("cot", "Reasoning → Key steps → Final code", "需要可讲解的解题过程"),
        ("self_consistency", "多路径采样 + 代码一致性投票", "降低偶然采样错误"),
        ("best_of_n", "N 个候选 + 语法/测试/格式重排", "有 assert 验证器时效果突出"),
        ("reflexion", "错误反馈 → 根因反思 → 重新生成", "调试失败代码"),
        ("tot", "思路扩展、评分、剪枝、再生成代码", "较复杂算法题"),
        ("hybrid", "ToT → SC → Best-of-N → Reflexion", "综合展示，质量优先"),
    ]
    for row in descriptions:
        table.add_row(*row)
    console.print(table)


def numbered_choice(console: Console, names: list[str], prompt: str, default: int = 1) -> str:
    for index, name in enumerate(names, 1):
        console.print(f"  [cyan]{index}[/cyan]. {name}")
    while True:
        selected = IntPrompt.ask(prompt, default=default)
        if 1 <= selected <= len(names):
            return names[selected - 1]
        console.print(f"[yellow]请输入 1 到 {len(names)} 之间的编号。[/yellow]")


def multiline_input(console: Console, title: str) -> str:
    console.print(f"[bold]{title}[/bold]")
    console.print("[dim]可输入一行或多行。完成后输入 .end，取消请输入 .cancel。[/dim]")
    lines: list[str] = []
    while True:
        line = console.input("[dim]│ [/dim]")
        if line.strip() == ".cancel":
            return ""
        if line.strip() == ".end":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def test_input(console: Console) -> list[str]:
    console.print("[bold]输入可选 assert 测试[/bold]")
    console.print("[dim]每行一条，例如 assert add(1, 2) == 3。完成后输入 .end。[/dim]")
    tests: list[str] = []
    while True:
        line = console.input("[dim]assert> [/dim]").strip()
        if line == ".end":
            break
        if line:
            if not line.startswith("assert "):
                console.print("[yellow]这不是 assert 语句，已跳过。[/yellow]")
                continue
            tests.append(line)
    return tests


def task_with_tests(task: str, tests: list[str]) -> str:
    if not tests:
        return task
    return f"{task}\n\nYour code should pass these tests:\n" + "\n".join(tests)


def best_path_messages(row: dict[str, Any], selected_plan: str = "") -> list[dict[str, str]]:
    """Match the successful prompt contract in best_path/evaluate.py."""
    tests = "\n".join(row["tests"])
    plan = f"\n\nSelected solution plan:\n{selected_plan}" if selected_plan else ""
    user = (
        "Solve the Python programming task below. Think briefly about edge cases, then return one complete solution "
        "inside a ```python code block. Do not use external input/output unless requested.\n\n"
        f"Task:\n{row['display_task']}\n\nVisible tests:\n{tests}{plan}"
    )
    return [
        {"role": "system", "content": "You are a precise Python code-generation assistant."},
        {"role": "user", "content": user},
    ]


def best_path_repair_messages(
    row: dict[str, Any], previous: dict[str, Any], round_id: int
) -> list[dict[str, str]]:
    details = []
    for index, result in enumerate(previous["test_results"]):
        if result["passed"]:
            continue
        test = row["tests"][index] if index < len(row["tests"]) else f"test {index + 1}"
        error = str(result.get("stderr") or result.get("error_type") or "failed")[-600:]
        details.append(f"Test: {test}\nFailure: {error}")
    base = best_path_messages(row)[1]["content"]
    user = (
        f"{base}\n\nThe following candidate is incorrect:\n```python\n{previous['final_code']}\n```\n\n"
        f"Reflexion round {round_id}: diagnose the failures and return a corrected complete solution "
        "in one ```python block. Do not return a patch.\n\n" + "\n\n".join(details[:3])
    )
    return [
        {"role": "system", "content": "You are a precise Python code-generation assistant."},
        {"role": "user", "content": user},
    ]


def build_row(task: str, tests: list[str]) -> dict[str, Any]:
    return {
        "index": 0,
        "task": task_with_tests(task, tests),
        "display_task": task,
        "reference": "",
        "test_setup": "",
        "tests": tests,
        "raw": {"instruction": task, "test_list": tests},
    }


def consensus_counts(candidates: list[dict[str, Any]]) -> Counter[str]:
    return Counter(canonical_code(item["final_code"]) for item in candidates)


def hybrid_score(candidate: dict[str, Any], agreement: Counter[str], total: int) -> float:
    consensus_bonus = 1.5 * agreement[canonical_code(candidate["final_code"])] / max(total, 1)
    return candidate["verifier_score"] + consensus_bonus


def run_hybrid(
    row: dict[str, Any], generator: ModelGenerator, args: argparse.Namespace, console: Console
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = generator.torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = generator.usage.snapshot()
    started = time.perf_counter()

    console.rule("[bold cyan]Stage 1/4 · Tree of Thoughts 规划搜索")
    beam = [{"text": "", "thought_score": 0.0, "depth": 0}]
    expanded_nodes = 0
    for depth in range(1, args.tree_depth + 1):
        requests = []
        owners = []
        for parent in beam:
            for branch in range(1, args.tree_width + 1):
                requests.append(thought_messages(row["task"], parent["text"], depth, branch))
                owners.append((parent, branch))
        outputs = generator.generate(
            requests,
            sample=True,
            max_new_tokens=min(args.max_new_tokens, 384),
            label=f"hybrid/tot depth {depth}",
        )
        expanded_nodes += len(outputs)
        children = []
        for output, (parent, branch) in zip(outputs, owners, strict=True):
            combined = f"{parent['text']}\n\n{output}".strip()
            children.append(
                {
                    "text": combined,
                    "thought_score": parent["thought_score"] + score_thought(output, row["task"], depth),
                    "depth": depth,
                    "branch": branch,
                }
            )
        beam = sorted(children, key=lambda item: item["thought_score"], reverse=True)[: args.tree_width]
    plan_table = Table(box=box.SIMPLE, show_header=True)
    plan_table.add_column("Path")
    plan_table.add_column("规划分", justify="right")
    plan_table.add_column("摘要")
    for index, plan in enumerate(beam, 1):
        summary = re.sub(r"\s+", " ", plan["text"])[:100]
        plan_table.add_row(str(index), f"{plan['thought_score']:.2f}", summary)
    console.print(plan_table)

    console.rule("[bold cyan]Stage 2/4 · Self-Consistency 多路径代码采样")
    requests = []
    plan_owners = []
    direct_count = max(1, (args.best_of_n + 1) // 2)
    for candidate_id in range(args.best_of_n):
        if candidate_id < direct_count:
            # Preserve multiple independent samples of the exact route used by
            # the formal best_path A5 run. Remaining requests use ToT plans.
            path_id = -1
            request = best_path_messages(row)
        else:
            path_id = (candidate_id - direct_count) % len(beam)
            request = best_path_messages(row, beam[path_id]["text"])
        request = append_instruction(
            request,
            f"This is independent candidate {candidate_id + 1}; verify the function signature and edge cases.",
        )
        requests.append(request)
        plan_owners.append(path_id)
    responses = generator.generate(requests, sample=True, label="hybrid/self-consistency")
    candidates = [
        evaluate_candidate(row, response, args, candidate_id)
        for candidate_id, response in enumerate(responses)
    ]
    agreement = consensus_counts(candidates)
    for candidate in candidates:
        candidate["plan_id"] = plan_owners[candidate["candidate_id"]]
        candidate["consensus_count"] = agreement[canonical_code(candidate["final_code"])]
        candidate["hybrid_score"] = hybrid_score(candidate, agreement, len(candidates))
    majority_count = max(agreement.values()) if agreement else 0
    console.print(
        f"生成 [bold]{len(candidates)}[/bold] 个候选；唯一规范化答案 {len(agreement)} 个；"
        f"最大一致票数 [bold]{majority_count}/{len(candidates)}[/bold]。"
    )

    console.rule("[bold cyan]Stage 3/4 · Best-of-N 执行验证与重排序")
    rank_table = Table(box=box.SIMPLE, show_header=True)
    rank_table.add_column("候选")
    rank_table.add_column("Plan")
    rank_table.add_column("Syntax")
    rank_table.add_column("Tests")
    rank_table.add_column("一致票")
    rank_table.add_column("综合分", justify="right")
    ranked = sorted(candidates, key=lambda item: item["hybrid_score"], reverse=True)
    for item in ranked:
        rank_table.add_row(
            str(item["candidate_id"] + 1),
            "direct CoT" if item["plan_id"] < 0 else str(item["plan_id"] + 1),
            "✓" if item["syntax_ok"] else "✗",
            f"{item['passed_tests']}/{item['total_tests']}" if item["total_tests"] else "n/a",
            str(item["consensus_count"]),
            f"{item['hybrid_score']:.2f}",
        )
    console.print(rank_table)
    selected = ranked[0]

    console.rule("[bold cyan]Stage 4/4 · Reflexion 失败反馈修正")
    history = [selected]
    should_reflect = not selected["syntax_ok"] or (selected["total_tests"] > 0 and not selected["passed"])
    for round_id in range(1, args.reflexion_rounds + 1):
        if not should_reflect:
            break
        console.print(f"[yellow]第 {round_id} 轮反馈：{feedback_from_case(history[-1])[:400]}[/yellow]")
        response = generator.generate(
            [best_path_repair_messages(row, history[-1], round_id)],
            sample=True,
            label=f"hybrid/reflexion round {round_id}",
        )[0]
        refined = evaluate_candidate(row, response, args, len(candidates) + round_id - 1)
        history.append(refined)
        should_reflect = not refined["syntax_ok"] or (refined["total_tests"] > 0 and not refined["passed"])
    if len(history) == 1:
        console.print("[green]最佳候选已通过验证，无需反思修正。[/green]")
    else:
        console.print(f"已完成 {len(history) - 1} 轮反思修正。")

    selected = select_best(candidates + history[1:])
    case = final_case(
        selected,
        "hybrid",
        selection="ToT planning + consistency bonus + verifier reranking + Reflexion",
        majority_consistency=majority_count / len(candidates),
        thought_paths=beam,
        candidates=[compact_candidate(item) for item in candidates],
        reflection_history=[compact_candidate(item) for item in history],
    )
    elapsed = time.perf_counter() - started
    usage = generator.usage.delta(before)
    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    metrics = summarize(
        "hybrid",
        [case],
        args,
        usage,
        elapsed,
        {
            "tree_width": args.tree_width,
            "tree_depth": args.tree_depth,
            "expanded_thought_nodes": expanded_nodes,
            "best_of_n": args.best_of_n,
            "num_samples": len(candidates),
            "reflexion_rounds": args.reflexion_rounds,
            "majority_consistency": majority_count / len(candidates),
        },
        peak_gpu_mb,
    )
    return case, metrics


def save_session(
    args: argparse.Namespace,
    task: str,
    tests: list[str],
    strategy: str,
    case: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = args.output_dir / f"{stamp}_{strategy}.json"
    save_json(
        path,
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "model_path": str(args.model_path),
            "strategy": strategy,
            "task": task,
            "tests": tests,
            "metrics": metrics,
            "result": case,
        },
    )
    return path


def display_result(
    console: Console,
    task: str,
    tests: list[str],
    strategy: str,
    case: dict[str, Any],
    metrics: dict[str, Any],
    saved_path: Path,
) -> None:
    console.rule("[bold green]最终结果")
    code = case.get("final_code", "")
    console.print(Syntax(code or "# No code extracted", "python", theme="monokai", line_numbers=True, word_wrap=True))
    table = Table(box=box.ROUNDED, border_style="green")
    table.add_column("策略", style="bold")
    table.add_column("语法")
    table.add_column("测试")
    table.add_column("Verifier")
    table.add_column("耗时")
    table.add_column("生成序列")
    table.add_column("生成 tokens")
    test_text = f"{case['passed_tests']}/{case['total_tests']}" if tests else "未提供"
    table.add_row(
        strategy,
        "✓" if case["syntax_ok"] else "✗",
        test_text,
        f"{case['verifier_score']:.2f}",
        f"{metrics['elapsed_seconds']:.2f}s",
        str(metrics["generated_sequences"]),
        str(metrics["generated_tokens"]),
    )
    console.print(table)
    if tests:
        if case["passed"]:
            console.print("[bold green]✓ 所有用户提供的测试均通过[/bold green]")
        else:
            console.print("[bold red]✗ 仍有测试未通过，请查看保存的执行反馈[/bold red]")
    console.print(f"[dim]本次完整轨迹已保存：{saved_path}[/dim]")


def ensure_model(generator: ModelGenerator | None, args: argparse.Namespace, console: Console) -> ModelGenerator:
    if generator is not None:
        return generator
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(
            f"模型目录不可用：{args.model_path}\n请运行 pathcoder doctor 检查环境，"
            "或通过 --model_path 指定有效模型。"
        )
    console.print(Panel("正在将 Qwen3 Full SFT 模型加载到 GPU。首次加载通常需要数秒。", border_style="cyan"))
    started = time.perf_counter()
    generator = ModelGenerator(args)
    generator.torch.manual_seed(args.seed)
    if generator.torch.cuda.is_available():
        generator.torch.cuda.manual_seed_all(args.seed)
    console.print(f"[bold green]模型已就绪[/bold green]  用时 {time.perf_counter() - started:.2f}s")
    return generator


def run_task(
    task: str,
    tests: list[str],
    strategy: str,
    generator: ModelGenerator,
    args: argparse.Namespace,
    console: Console,
) -> None:
    row = build_row(task, tests)
    console.print(Panel(task, title="任务", border_style="blue"))
    if tests:
        console.print(Panel("\n".join(tests), title=f"用户验证器 · {len(tests)} tests", border_style="yellow"))
    else:
        console.print("[yellow]未提供测试：将只使用语法、格式和一致性信号进行选择。[/yellow]")
    if strategy == "hybrid":
        case, metrics = run_hybrid(row, generator, args, console)
    else:
        cases, metrics = run_strategy(strategy, [row], generator, args)
        case = cases[0]
    path = save_session(args, task, tests, strategy, case, metrics)
    display_result(console, task, tests, strategy, case, metrics, path)


def select_example(console: Console) -> tuple[str, list[str]]:
    keys = list(EXAMPLES)
    labels = [f"{EXAMPLES[key]['title']} ({key})" for key in keys]
    selected_label = numbered_choice(console, labels, "选择示例", default=1)
    key = keys[labels.index(selected_label)]
    return EXAMPLES[key]["task"], list(EXAMPLES[key]["tests"])


def select_strategy(console: Console, default: str = "hybrid") -> str:
    labels = {
        "greedy": "greedy · 极速单次生成",
        "cot": "cot · 可解释链式推理",
        "self_consistency": "self_consistency · 多路径投票",
        "best_of_n": "best_of_n · 多候选验证重排",
        "reflexion": "reflexion · 执行反馈修正",
        "tot": "tot · 思路树搜索",
        "hybrid": "hybrid · 四策略综合（推荐）",
    }
    keys = list(labels)
    default_index = keys.index(default) + 1
    selected = numbered_choice(console, [labels[key] for key in keys], "选择推理策略", default=default_index)
    return keys[list(labels.values()).index(selected)]


def interactive(args: argparse.Namespace, console: Console) -> None:
    generator: ModelGenerator | None = None
    banner(console)
    show_home(console, args)
    try:
        while True:
            console.print()
            menu = [
                "运行内置示例",
                "输入自己的编程任务",
                "查看最近会话",
                "查看 best_path 核心性能",
                "查看全项目训练与评估指标",
                "效果分析与原因解析",
                "查看策略说明",
                "检查运行环境",
                "退出",
            ]
            action = numbered_choice(console, menu, "请选择操作", default=1)
            if action == "退出":
                break
            if action == "查看 best_path 核心性能":
                show_performance(console, args)
                continue
            if action == "查看最近会话":
                show_history(console, args)
                continue
            if action == "查看全项目训练与评估指标":
                show_all_metrics(console, args)
                continue
            if action == "效果分析与原因解析":
                show_effect_analysis(console, args)
                continue
            if action == "查看策略说明":
                show_strategies(console)
                continue
            if action == "检查运行环境":
                show_doctor(console, args)
                continue
            if action == "运行内置示例":
                show_examples(console)
                task, tests = select_example(console)
            else:
                task = multiline_input(console, "输入 Python 编程任务")
                if not task:
                    console.print("[yellow]任务为空，返回主菜单。[/yellow]")
                    continue
                tests = test_input(console)
            strategy = select_strategy(console, args.strategy)
            try:
                generator = ensure_model(generator, args, console)
                run_task(task, tests, strategy, generator, args, console)
            except (FileNotFoundError, RuntimeError) as exc:
                console.print(Panel(str(exc), title="无法运行任务", border_style="red"))
                continue
            if not Confirm.ask("继续体验其他任务？", default=True):
                break
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]已结束交互。[/yellow]")
    finally:
        if generator is not None:
            generator.close()
    console.print("[bold cyan]PathCoder 已退出。会话记录已保留。[/bold cyan]")


def main() -> None:
    args = parse_args()
    console = make_console(args)
    random.seed(args.seed)
    if args.list_examples:
        banner(console)
        show_examples(console)
        return
    if args.show_strategies:
        banner(console)
        show_strategies(console)
        return
    if args.show_history:
        banner(console)
        show_history(console, args)
        return
    if args.doctor:
        banner(console)
        if not show_doctor(console, args):
            raise SystemExit(1)
        return
    if args.export_catalog:
        records = discover_metric_records(PROJECT_ROOT)
        save_json(args.export_catalog, export_catalog(records))
        console.print(f"[green]已导出 {len(records)} 条规范化指标：{args.export_catalog}[/green]")
        if not (args.show_all_metrics or args.analyze_results or args.show_performance or args.task or args.example):
            return
    if args.show_all_metrics and not (args.task or args.example):
        banner(console)
        show_all_metrics(console, args)
        return
    if args.analyze_results and not (args.task or args.example):
        banner(console)
        show_effect_analysis(console, args)
        return
    if args.show_performance and not (args.task or args.example):
        banner(console)
        show_performance(console, args)
        return
    if not (args.task or args.example):
        interactive(args, console)
        return

    banner(console)
    if args.example:
        example = EXAMPLES[args.example]
        task = example["task"]
        tests = list(example["tests"])
    else:
        task = args.task.strip()
        tests = list(args.test)
    try:
        generator = ensure_model(None, args, console)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(Panel(str(exc), title="无法加载模型", border_style="red"))
        raise SystemExit(1) from exc
    try:
        run_task(task, tests, args.strategy, generator, args, console)
    finally:
        generator.close()


if __name__ == "__main__":
    main()
