#!/usr/bin/env python3
"""Render a run-specific A2-full PPO config from a stable YAML template."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def maybe_set(config: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        config[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reward-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="a2full_reward_ppo_train")
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--ppo-target", type=float, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML object in {args.template}")

    config["model_name_or_path"] = args.model_path
    config["ref_model"] = args.model_path
    config["reward_model"] = args.reward_url
    config["dataset"] = args.dataset
    config["output_dir"] = args.output_dir
    maybe_set(config, "num_train_epochs", args.num_train_epochs)
    maybe_set(config, "max_steps", args.max_steps if args.max_steps and args.max_steps > 0 else None)
    maybe_set(config, "learning_rate", args.learning_rate)
    maybe_set(config, "ppo_target", args.ppo_target)
    maybe_set(config, "save_steps", args.save_steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Rendered PPO config: {args.output}")


if __name__ == "__main__":
    main()
