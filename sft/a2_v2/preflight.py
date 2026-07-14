#!/usr/bin/env python3
"""Validate the CUDA 13 bitsandbytes and QLoRA/QDoRA runtime."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch


def run_variant(model_path: Path, use_dora: bool) -> dict:
    from bitsandbytes.nn import Linear4bit
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    linear4_count = sum(isinstance(module, Linear4bit) for module in model.modules())
    if linear4_count == 0:
        raise RuntimeError("No bitsandbytes Linear4bit modules were created")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules="all-linear",
            use_dora=use_dora,
        ),
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Write a Python function add(a, b)."},
         {"role": "assistant", "content": "def add(a, b):\n    return a + b"}],
        tokenize=False,
        enable_thinking=False,
    )
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    outputs = model(**tokens, labels=tokens["input_ids"])
    loss = float(outputs.loss.detach().cpu())
    if not math.isfinite(loss):
        raise RuntimeError(f"Non-finite {'QDoRA' if use_dora else 'QLoRA'} loss: {loss}")
    outputs.loss.backward()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gradients = sum(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    if not trainable or not gradients:
        raise RuntimeError("No trainable adapter gradients were produced")
    result = {
        "variant": "qdora" if use_dora else "qlora",
        "linear4_modules": linear4_count,
        "loss": loss,
        "trainable_parameters": trainable,
        "parameters_with_gradient": gradients,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    del outputs, model, tokenizer, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/root/project/Qwen3-0.6B"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true", help="Only validate native import and GPU visibility.")
    args = parser.parse_args()

    import bitsandbytes as bnb

    report = {
        "bitsandbytes": bnb.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "variants": [],
    }
    if not report["cuda_available"]:
        raise RuntimeError("CUDA is unavailable")
    if not args.quick:
        report["variants"].append(run_variant(args.model_path, use_dora=False))
        report["variants"].append(run_variant(args.model_path, use_dora=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
