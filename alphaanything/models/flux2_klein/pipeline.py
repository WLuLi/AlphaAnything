"""FLUX.2 Klein RGBA inference pipeline."""
from __future__ import annotations

import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from alphaanything.checkpoint import load_adapter_bundle
from alphaanything.modules.alpha_math import ea_score_to_alpha
from . import model_i2rgba, model_t2rgba
from .vae import SinglePassRGBAFlux2VAE


class FullCanvasFlux2ImageProcessor(Flux2ImageProcessor):
    def _resize_and_crop(self, image, width: int, height: int):
        return self.resize(image, height=height, width=width, resize_mode="default")


def load_model(config, checkpoint, device, dtype):
    pipe = Flux2KleinPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        local_files_only=bool(config["model"].get("local_files_only", False)),
    )
    if config["task"] == "image_to_rgba":
        pipe.image_processor = FullCanvasFlux2ImageProcessor.from_config(pipe.image_processor.config)
    rgba_vae = SinglePassRGBAFlux2VAE(pipe.vae, config["vae_alpha_adapter"])
    load_adapter_bundle(pipe, rgba_vae, checkpoint, config)
    pipe.to(device)
    rgba_vae.to(device=device, dtype=dtype).eval()
    return pipe, rgba_vae


@torch.inference_mode()
def generate(
    pipe, rgba_vae, *, prompt, image=None, height=1024, width=1024,
    num_inference_steps=50, guidance_scale=4.0, negative_prompt=" ", generator, config,
):
    args = (
        prompt, height, width, num_inference_steps, guidance_scale, negative_prompt,
        generator, pipe._execution_device, config["model"],
    )
    if config["task"] == "image_to_rgba":
        latent = model_i2rgba.sample(pipe, image, *args)
    else:
        latent = model_t2rgba.sample(pipe, *args)
    rgb, score = rgba_vae(latent, operation="decode_rgba")
    return rgb[0], ea_score_to_alpha(score)[0]
