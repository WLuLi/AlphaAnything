"""Qwen-Image-2512 text-to-RGBA inference pipeline."""
from __future__ import annotations

import torch

from alphaanything.checkpoint import load_adapter_bundle
from alphaanything.models.qwen_common.vae import SinglePassRGBAQwenVAE
from alphaanything.modules.alpha_math import ea_score_to_alpha
from .model import load_pipeline, sample_latents


def load_model(config, checkpoint, device, dtype):
    pipe = load_pipeline(config["model"], dtype)
    rgba_vae = SinglePassRGBAQwenVAE(pipe.vae, config["vae_alpha_adapter"])
    load_adapter_bundle(pipe, rgba_vae, checkpoint, config)
    pipe.to(device=device)
    pipe.transformer.requires_grad_(False).eval()
    pipe.set_progress_bar_config(disable=True)
    rgba_vae.to(device=device, dtype=dtype).eval()
    return pipe, rgba_vae


@torch.inference_mode()
def generate(
    pipe, rgba_vae, *, prompt, image=None, height=1024, width=1024,
    num_inference_steps=50, guidance_scale=4.0, negative_prompt=" ", generator, config,
):
    latents = sample_latents(
        pipe, [prompt], height=height, width=width, steps=num_inference_steps,
        guidance_scale=guidance_scale, negative_prompt=negative_prompt,
        generator=generator, model=config["model"],
    )
    rgb, score = rgba_vae(latents, operation="decode_rgba")
    return rgb[0], ea_score_to_alpha(score)[0]
