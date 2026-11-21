"""Qwen-Image-2512 text-to-RGBA model operations."""
from __future__ import annotations

import torch

from alphaanything.models.qwen_common.components import unpack_qwen_latents
from alphaanything.utils.model import isolated_scheduler

DEFAULT_MODEL = "Qwen/Qwen-Image-2512"


def load_pipeline(model, dtype):
    from diffusers import QwenImagePipeline

    kwargs = {"torch_dtype": dtype, "local_files_only": bool(model.get("local_files_only", False))}
    if model.get("revision"):
        kwargs["revision"] = model["revision"]
    return QwenImagePipeline.from_pretrained(model.get("pretrained_model_name_or_path", DEFAULT_MODEL), **kwargs)


@torch.no_grad()
def sample_latents(
    pipe, prompts, *, height, width, steps, guidance_scale, negative_prompt, generator, model,
):
    with isolated_scheduler(pipe):
        tokens = pipe(
            prompt=prompts,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            true_cfg_scale=guidance_scale,
            generator=generator,
            max_sequence_length=int(model.get("max_sequence_length", 512)),
            output_type="latent",
            return_dict=False,
        )[0]
    return unpack_qwen_latents(tokens, height=height, width=width, vae_scale_factor=int(pipe.vae_scale_factor))
