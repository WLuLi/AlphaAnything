"""Portable inference adapters: transformer LoRA, alpha branch, and model metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LORA_FILENAME = "pytorch_lora_weights.safetensors"
ALPHA_FILENAME = "alpha_decoder.safetensors"
CONFIG_FILENAME = "adapter_config.json"
TASKS = {
    "flux2_klein": {"text_to_rgba", "image_to_rgba"},
    "qwen_image": {"text_to_rgba"},
    "qwen_image_edit": {"image_to_rgba"},
}
_MODEL_EXPORT_KEYS = ("pretrained_model_name_or_path", "max_sequence_length", "text_encoder_out_layers", "condition_image_area")
_VAE_EXPORT_KEYS = (
    "stage_channels", "injection_rank", "injection_alpha", "injection_stage_gates",
    "res_blocks_per_stage", "attention_heads", "head_channels",
)


def read_adapter_config(checkpoint: str | Path) -> dict[str, Any]:
    directory = Path(checkpoint)
    path = directory / CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}.")
    config = json.loads(path.read_text(encoding="utf-8"))
    backbone, task = config.get("backbone"), config.get("task")
    if backbone not in TASKS or task not in TASKS[backbone]:
        raise ValueError(f"Unsupported backbone/task combination: {backbone}/{task}.")
    for name in (LORA_FILENAME, ALPHA_FILENAME):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return config


def load_adapter_bundle(pipe, rgba_vae, checkpoint: str | Path, config: dict[str, Any]) -> None:
    """Load LoRA and alpha branch weights and match the configured LoRA alpha/rank scale."""
    from safetensors.torch import load_file
    from alphaanything.utils.lora import lora_adapter_weight_for_config

    lora_cfg = config["lora"]
    adapter_name = str(lora_cfg.get("adapter_name", "alphaanything"))
    pipe.load_lora_weights(str(checkpoint), weight_name=LORA_FILENAME, adapter_name=adapter_name)
    scale = lora_adapter_weight_for_config(pipe.transformer, adapter_name, lora_cfg)
    pipe.set_adapters([adapter_name], adapter_weights=[scale])
    rgba_vae.load_adapter_state_dict(load_file(str(Path(checkpoint) / ALPHA_FILENAME), device="cpu"))
    pipe.transformer.requires_grad_(False).eval()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.requires_grad_(False).eval()
    rgba_vae.requires_grad_(False).eval()


def adapter_metadata(config: dict[str, Any], *, backbone: str, task: str, step: int) -> dict[str, Any]:
    """Build portable adapter metadata stripped of training-only and machine-specific fields."""
    if backbone not in TASKS or task not in TASKS[backbone]:
        raise ValueError(f"Unsupported backbone/task combination: {backbone}/{task}.")
    source_model = config["model"]
    base_model = source_model["pretrained_model_name_or_path"]

    model = {k: source_model[k] for k in _MODEL_EXPORT_KEYS if k in source_model}
    model.setdefault("max_sequence_length", 512)
    if backbone == "qwen_image_edit":
        model.setdefault("condition_image_area", 384 * 384)

    source_lora = config.get("lora") or {}
    lora = {
        "adapter_name": source_lora.get("adapter_name", source_model.get("adapter_name", f"{backbone}_{task}")),
        "rank": int(source_lora.get("rank", source_model.get("lora_rank", 64))),
        "alpha": int(source_lora.get("alpha", source_model.get("lora_alpha", 128))),
    }
    vae_cfg = config["vae_alpha_adapter"]
    vae_alpha_adapter = {k: vae_cfg[k] for k in _VAE_EXPORT_KEYS if k in vae_cfg}

    evaluation = config.get("evaluation", {})
    guidance = evaluation.get("true_cfg_scale", evaluation.get("guidance_scale", 4.0))
    inference = {
        "height": 1024,
        "width": 1024,
        "num_inference_steps": int(evaluation.get("inference_steps", 50)),
        "guidance_scale": float(4.0 if guidance is None else guidance),
        "negative_prompt": "" if (backbone == "flux2_klein" and task == "image_to_rgba") else str(evaluation.get("negative_prompt", " ")),
        "seed": 42,
        "mixed_precision": "bf16",
    }
    return {
        "format_version": 1,
        "backbone": backbone,
        "task": task,
        "base_model": base_model,
        "training_step": int(step),
        "model": model,
        "lora": lora,
        "vae_alpha_adapter": vae_alpha_adapter,
        "inference": inference,
    }
