"""Task-specific FLUX.2 Klein image-to-RGBA operations."""
from __future__ import annotations

from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from .conditioning import flux2_model_inputs, sample_flux2_rgba_latents, training_text_features

TASK = "image_to_rgba"
TARGET_KEY = "target_pixel_values"


def encode_condition(
    raw_vae: torch.nn.Module, batch: dict[str, Any], *, device: torch.device, weight_dtype: torch.dtype,
) -> torch.Tensor:
    condition = batch["condition_pixel_values"].to(device=device, dtype=weight_dtype, non_blocking=True)
    return raw_vae.encode_normalized_mode(condition)


encode_text_features = training_text_features


def forward(
    pipe: Flux2KleinPipeline, noisy_target: torch.Tensor, condition: torch.Tensor,
    timesteps: torch.Tensor, embeds: torch.Tensor, text_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, image_ids, target_ids = flux2_model_inputs(noisy_target, condition)
    output = pipe.transformer(
        hidden_states=inputs, timestep=timesteps / 1000.0, guidance=None,
        encoder_hidden_states=embeds, txt_ids=text_ids, img_ids=image_ids, return_dict=False,
    )[0]
    flow = Flux2KleinPipeline._unpack_latents_with_ids(output[:, :target_ids.shape[1]], target_ids)
    return flow, target_ids


@torch.no_grad()
def sample(
    pipe: Flux2KleinPipeline, image: Image.Image, prompt: str | list[str], height: int, width: int,
    num_inference_steps: int, guidance_scale: float, negative_prompt: str,
    generator: torch.Generator | list[torch.Generator], device: torch.device, model_cfg: dict[str, Any],
) -> torch.Tensor:
    return sample_flux2_rgba_latents(
        pipe, image, prompt, height, width, num_inference_steps, guidance_scale,
        negative_prompt, generator, device, model_cfg,
    )
