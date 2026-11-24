"""Qwen-Image-Edit-2511 image-to-RGBA inference pipeline."""
from __future__ import annotations

import torch
from torchvision.transforms.functional import to_tensor

from alphaanything.checkpoint import load_adapter_bundle
from alphaanything.models.qwen_common.vae import SinglePassRGBAQwenVAE
from .model import load_pipeline, sample_qwen_edit_rgba


def load_base_pipeline(config, device, dtype):
    pipe = load_pipeline(config["model"], dtype)
    pipe.to(device=device)
    pipe.transformer.requires_grad_(False).eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def create_rgba_vae(pipe, vae_config, device, dtype):
    return SinglePassRGBAQwenVAE(pipe.vae, vae_config).to(device=device, dtype=dtype).eval()


def load_model(config, checkpoint, device, dtype):
    pipe = load_base_pipeline(config, device, dtype)
    rgba_vae = create_rgba_vae(pipe, config["vae_alpha_adapter"], device, dtype)
    load_adapter_bundle(pipe, rgba_vae, checkpoint, config)
    pipe.to(device=device)
    pipe.transformer.requires_grad_(False).eval()
    rgba_vae.to(device=device, dtype=dtype).eval()
    return pipe, rgba_vae


@torch.inference_mode()
def generate(
    pipe, rgba_vae, *, prompt, image=None, height=1024, width=1024,
    num_inference_steps=50, guidance_scale=4.0, negative_prompt=" ", generator, config,
    enable_alpha: bool = True,
):
    condition = to_tensor(image.convert("RGB")) * 2.0 - 1.0
    example = {"condition_pixel_values": condition, "prompt": prompt}
    inference = {
        **config.get("inference", {}),
        "num_inference_steps": int(num_inference_steps),
        "true_cfg_scale": float(guidance_scale),
        "negative_prompt": str(negative_prompt),
    }
    param = next(pipe.transformer.parameters())
    device, dtype = param.device, param.dtype
    with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
        rgb, alpha = sample_qwen_edit_rgba(
            example=example, config={**config, "inference": inference},
            transformer=pipe.transformer, rgba_vae=rgba_vae, scheduler=pipe.scheduler,
            prompt_pipeline=pipe, device=device, weight_dtype=dtype, generator=generator,
            enable_alpha=enable_alpha,
        )
    return rgb[0], (alpha[0] if alpha is not None else None)
