#!/usr/bin/env python3
"""Render a LLaMA-Factory YAML config with restart-safe overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-mode", choices=("trainer", "model", "adapter", "none"), default="none")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--epochs-already", type=float, default=0.0)
    args = parser.parse_args()

    data = yaml.safe_load(args.template.read_text(encoding="utf-8"))

    if args.resume_mode == "trainer" and args.resume_checkpoint:
        data["resume_from_checkpoint"] = args.resume_checkpoint
    elif args.resume_mode == "model" and args.resume_checkpoint:
        data["model_name_or_path"] = args.resume_checkpoint
        data["resume_from_checkpoint"] = None
    elif args.resume_mode == "adapter" and args.resume_checkpoint:
        data["adapter_name_or_path"] = args.resume_checkpoint
        data["resume_from_checkpoint"] = None
    elif args.resume_mode == "adapter" and args.adapter_path:
        data["adapter_name_or_path"] = args.adapter_path
        data["resume_from_checkpoint"] = None
    else:
        data["resume_from_checkpoint"] = None

    if args.resume_mode in {"model", "adapter"}:
        data["resume_from_checkpoint"] = False
        if args.epochs_already > 0 and "num_train_epochs" in data:
            remaining_epochs = max(0.01, float(data["num_train_epochs"]) - args.epochs_already)
            data["num_train_epochs"] = round(remaining_epochs, 6)

    if data.get("stage") == "ppo":
        data.pop("resume_from_checkpoint", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
