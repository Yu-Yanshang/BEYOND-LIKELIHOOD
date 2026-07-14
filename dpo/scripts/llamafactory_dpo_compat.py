#!/usr/bin/env python3
"""DPO-local LLaMA-Factory launcher for runtime compatibility patches."""

from __future__ import annotations


def install_patch() -> None:
    # Reuse the project-approved A2 launcher fixes without editing vendored code.
    from sft.a2_v2.llamafactory_compat import install_patch as install_a2_patch

    install_a2_patch()

    # TRL 0.24 imports this private Transformers mapping. The local
    # Transformers 5 build no longer exposes it. DPO here is text-only Qwen3, so
    # an empty compatibility mapping is sufficient for import-time checks.
    import transformers.models.auto.modeling_auto as modeling_auto

    if not hasattr(modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
        modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = {}

    import trl.models.utils as trl_model_utils

    if not hasattr(trl_model_utils, "prepare_deepspeed"):
        trl_model_utils.prepare_deepspeed = lambda model, *args, **kwargs: model
    if not hasattr(trl_model_utils, "prepare_fsdp"):
        trl_model_utils.prepare_fsdp = lambda model, *args, **kwargs: model


def main() -> None:
    install_patch()
    from llamafactory.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
