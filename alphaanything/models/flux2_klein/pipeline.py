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


def load_base_pipeline(config, device, dtype):
    pipe = Flux2KleinPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        torch_dtype=dtype,
        local_files_only=bool(config["model"].get("local_files_only", False)),
    )
    pipe.image_processor = FullCanvasFlux2ImageProcessor.from_config(pipe.image_processor.config)
    pipe.to(device)
    pipe.transformer.requires_grad_(False).eval()
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.requires_grad_(False).eval()
    return pipe


def create_rgba_vae(pipe, vae_config, device, dtype):
    return SinglePassRGBAFlux2VAE(pipe.vae, vae_config).to(device=device, dtype=dtype).eval()


def load_model(config, checkpoint, device, dtype):
    pipe = load_base_pipeline(config, device, dtype)
    rgba_vae = create_rgba_vae(pipe, config["vae_alpha_adapter"], device, dtype)
    load_adapter_bundle(pipe, rgba_vae, checkpoint, config)
    pipe.to(device)
    rgba_vae.to(device=device, dtype=dtype).eval()
    return pipe, rgba_vae


@torch.inference_mode()
def generate(
    pipe, rgba_vae, *, prompt, image=None, height=1024, width=1024,
    num_inference_steps=50, guidance_scale=4.0, negative_prompt=" ", generator, config,
    enable_alpha: bool = True,
):
    args = (
        prompt, height, width, num_inference_steps, guidance_scale, negative_prompt,
        generator, pipe._execution_device, config["model"],
    )
    if image is not None:
        latent = model_i2rgba.sample(pipe, image, *args)
    else:
        latent = model_t2rgba.sample(pipe, *args)
    if enable_alpha:
        rgb, score = rgba_vae(latent, operation="decode_rgba")
        return rgb[0], ea_score_to_alpha(score)[0]
    rgb = rgba_vae(latent, operation="decode_rgb")
    return rgb[0], None
