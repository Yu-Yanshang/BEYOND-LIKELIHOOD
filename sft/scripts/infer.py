#!/usr/bin/env python3
"""Run single-prompt or batch inference with the fine-tuned Qwen model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "sft" / "outputs" / "qwen3_06b_code_full_sft"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "sft" / "outputs" / "infer_results.jsonl"


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return [data]
    return [{"prompt": block.strip()} for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]


def row_to_prompt(row: Any) -> str:
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return str(row).strip()
    if row.get("prompt"):
        return str(row["prompt"]).strip()

    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def generate_response(model, tokenizer, prompt: str, args) -> str:
    import torch

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    response_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompts_file", type=Path, default=None)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant that writes correct Python code.")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    if not args.prompt and args.prompts_file is None:
        raise ValueError("Provide --prompt or --prompts_file.")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing model directory: {args.model}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    rows = [{"prompt": args.prompt}] if args.prompt else load_prompts(args.prompts_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            prompt = row_to_prompt(row)
            response = generate_response(model, tokenizer, prompt, args)
            result = {"index": index, "prompt": prompt, "response": response}
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False))

    print(f"Wrote inference results to {args.output_file}")


if __name__ == "__main__":
    main()
