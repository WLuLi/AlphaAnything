from __future__ import annotations

from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from peft import LoraConfig


def flux2_lora_target_modules(transformer: torch.nn.Module, model_cfg: dict[str, Any]) -> list[str]:
    targets = list(model_cfg.get("lora_target_modules") or ["to_q", "to_k", "to_v", "to_out.0", "to_qkv_mlp_proj"])
    config = transformer.config if hasattr(transformer, "config") else transformer.module.config
    num_single_layers = int(getattr(config, "num_single_layers", 0))
    targets.extend(f"single_transformer_blocks.{i}.attn.to_out" for i in range(num_single_layers))
    return list(dict.fromkeys(targets))


def configure_pipeline_for_lora(
    pipe: Flux2KleinPipeline, model_cfg: dict[str, Any], train_cfg: dict[str, Any], adapter_name: str,
) -> None:
    for name in ("vae", "text_encoder"):
        module = getattr(pipe, name, None)
        if module is not None:
            module.requires_grad_(False).eval()
    pipe.transformer.requires_grad_(False)
    pipe.transformer.add_adapter(
        LoraConfig(
            r=int(model_cfg.get("lora_rank", 64)),
            lora_alpha=float(model_cfg.get("lora_alpha", 128)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
            init_lora_weights=model_cfg.get("lora_init_weights", "gaussian"),
            target_modules=flux2_lora_target_modules(pipe.transformer, model_cfg),
        ),
        adapter_name=adapter_name,
    )
    pipe.transformer.set_adapter(adapter_name)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        pipe.transformer.enable_gradient_checkpointing()
