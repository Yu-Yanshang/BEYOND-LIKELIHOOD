#!/usr/bin/env python3
"""A2-local LLaMA-Factory launcher with Transformers 5 BNB enum compatibility.

Transformers 5 exposes ``QuantizationMethod.BITS_AND_BYTES`` (value
``bitsandbytes``), while this local LLaMA-Factory revision compares it with its
own ``QuantizationMethod.BNB`` (value ``bnb``).  The mismatch incorrectly
rejects QDoRA before PEFT is initialized.  This launcher normalizes only that
runtime attribute and leaves the vendored LLaMA-Factory source untouched.
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path


def install_patch() -> None:
    # Transformers 5 changed its private _is_package_available helper to return
    # (available, version). TRL 0.24 stores that tuple in boolean flags, making
    # missing optional packages such as mergekit look truthy during import.
    import trl.import_utils as trl_import_utils

    for name, value in vars(trl_import_utils).items():
        if name.startswith("_") and name.endswith("_available") and isinstance(value, tuple):
            setattr(trl_import_utils, name, bool(value[0]))

    from llamafactory.extras.constants import QuantizationMethod
    from llamafactory.model import adapter

    original = adapter._setup_lora_tuning
    if getattr(original, "_a2_bnb_compat", False):
        return

    def compatible_setup(config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32):
        method = getattr(model, "quantization_method", None)
        method_value = str(getattr(method, "value", method)).lower()
        if finetuning_args.use_dora and method_value in {"bitsandbytes", "bnb"}:
            model.quantization_method = QuantizationMethod.BNB
        return original(config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32)

    compatible_setup._a2_bnb_compat = True
    adapter._setup_lora_tuning = compatible_setup


def main() -> None:
    telemetry_path = os.environ.get("A2_TORCH_TELEMETRY")
    if telemetry_path:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def save_telemetry() -> None:
            if not torch.cuda.is_available():
                return
            path = Path(telemetry_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        atexit.register(save_telemetry)
    install_patch()
    from llamafactory.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
