from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def lora_adapter_metadata(model: Any, adapter_name: str) -> dict[str, Any]:
    """Return the PEFT config dict needed to preserve alpha/rank on Diffusers reload."""
    return model.peft_config[adapter_name].to_dict()


def _lora_scale(rank: float, alpha: float, use_rslora: bool) -> float:
    denom = math.sqrt(rank) if use_rslora else rank
    return alpha / denom


def lora_adapter_weight_for_config(
    model: Any,
    adapter_name: str,
    lora_config: Mapping[str, Any],
) -> float:
    """Match the configured LoRA alpha/rank scale after Diffusers reload."""
    target = _lora_scale(
        float(lora_config.get("rank", 64)),
        float(lora_config.get("alpha", 128)),
        bool(lora_config.get("use_rslora", False)),
    )
    loaded_cfg = model.peft_config[adapter_name]
    loaded = _lora_scale(
        float(loaded_cfg.r),
        float(loaded_cfg.lora_alpha),
        bool(getattr(loaded_cfg, "use_rslora", False)),
    )
    return target / loaded
