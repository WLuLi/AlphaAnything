from __future__ import annotations

from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from alphaanything.utils.model import isolated_scheduler


def flux2_model_inputs(
    noisy_target: torch.Tensor, condition: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack target tokens, optionally appending a clean per-sample reference at T=10."""
    target_ids = Flux2KleinPipeline._prepare_latent_ids(noisy_target).to(noisy_target.device)
    target_tokens = Flux2KleinPipeline._pack_latents(noisy_target)
    if condition is None:
        return target_tokens, target_ids, target_ids
    condition_ids = Flux2KleinPipeline._prepare_image_ids([condition[:1]]).to(condition.device)
    condition_ids = condition_ids.expand(condition.shape[0], -1, -1)
    tokens = torch.cat([target_tokens, Flux2KleinPipeline._pack_latents(condition)], dim=1)
    return tokens, torch.cat([target_ids, condition_ids], dim=1), target_ids


def latent_ids_for_image_tokens(
    pipe: Flux2KleinPipeline, tokens: torch.Tensor, height: int, width: int,
) -> torch.Tensor:
    factor = int(pipe.vae_scale_factor) * 2
    shape = (tokens.shape[0], tokens.shape[-1], int(height) // factor, int(width) // factor)
    dummy = torch.empty(shape, device=tokens.device, dtype=tokens.dtype)
    return pipe._prepare_latent_ids(dummy).to(device=tokens.device)


@torch.no_grad()
def training_text_features(
    pipe: Flux2KleinPipeline,
    prompts: list[str],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = float(training_config.get("text_dropout_probability", 0.0))
    dropped = torch.rand(len(prompts)) < probability
    null_prompt = str(training_config.get("text_dropout_null_prompt", " "))
    selected = [null_prompt if drop else prompt for prompt, drop in zip(prompts, dropped.tolist(), strict=True)]
    embeddings, ids = pipe.encode_prompt(
        prompt=selected,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=int(model_config.get("max_sequence_length", 512)),
        text_encoder_out_layers=tuple(model_config.get("text_encoder_out_layers", [9, 18, 27])),
    )
    return embeddings, ids, dropped.float().mean().to(device)


@torch.no_grad()
def sample_flux2_rgba_latents(
    pipe: Flux2KleinPipeline,
    image: Image.Image | None,
    prompt: str | list[str],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    negative_prompt: str,
    generator: torch.Generator | list[torch.Generator],
    device: torch.device,
    model_cfg: dict[str, Any],
) -> torch.Tensor:
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    layers = tuple(model_cfg.get("text_encoder_out_layers", [9, 18, 27]))
    max_length = int(model_cfg.get("max_sequence_length", 512))
    negative_embeddings = None
    if guidance_scale > 1.0 and negative_prompt:
        negative_embeddings, _ = pipe.encode_prompt(
            prompt=[negative_prompt] * len(prompts), device=device, num_images_per_prompt=1,
            max_sequence_length=max_length, text_encoder_out_layers=layers,
        )
    captured: dict[str, torch.Tensor] = {}

    def capture(_pipe, _step, _timestep, kwargs):
        captured["tokens"] = kwargs["latents"].detach().clone()
        return kwargs

    with isolated_scheduler(pipe):
        pipe(
            image=image, prompt=prompts, negative_prompt_embeds=negative_embeddings,
            height=int(height), width=int(width), num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale), num_images_per_prompt=1, generator=generator,
            output_type="latent", return_dict=False, callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"], max_sequence_length=max_length,
            text_encoder_out_layers=layers,
        )
    tokens = captured["tokens"]
    ids = latent_ids_for_image_tokens(pipe, tokens, height, width)
    return Flux2KleinPipeline._unpack_latents_with_ids(tokens, ids)
