"""Qwen-Image-Edit-2511 latent sampling and RGBA decoding."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from alphaanything.models.qwen_common.components import (
    normalized_vae_mode,
    pack_qwen_latents,
    pad_prompt_embeddings,
    unpack_qwen_latents,
)
from alphaanything.modules.alpha_math import ea_score_to_alpha

DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"


def load_pipeline(model, dtype):
    from diffusers import QwenImageEditPlusPipeline

    kwargs = {"torch_dtype": dtype, "local_files_only": bool(model.get("local_files_only", False))}
    if model.get("revision"):
        kwargs["revision"] = model["revision"]
    return QwenImageEditPlusPipeline.from_pretrained(model.get("pretrained_model_name_or_path", DEFAULT_MODEL), **kwargs)


def _condition_tensor_to_vlm_image(condition_tensor: torch.Tensor, target_area: int) -> Image.Image:
    image = to_pil_image(((condition_tensor.detach().float().cpu() + 1.0) * 0.5).clamp(0.0, 1.0))
    ratio = float(image.width) / float(image.height)
    tw = max(32, int(round(math.sqrt(target_area * ratio) / 32)) * 32)
    th = max(32, int(round(math.sqrt(target_area / ratio) / 32)) * 32)
    return image.resize((tw, th), resample=Image.Resampling.LANCZOS)


@torch.no_grad()
def _encode_prompt(prompt_pipeline, condition_tensor, prompt, *, device, max_sequence_length, condition_image_area):
    vlm_img = _condition_tensor_to_vlm_image(condition_tensor, condition_image_area)
    embeds, mask = prompt_pipeline.encode_prompt(
        image=[vlm_img], prompt=str(prompt), device=device,
        num_images_per_prompt=1, max_sequence_length=int(max_sequence_length),
    )
    emb = embeds[0]
    if mask is not None:
        emb = emb[: int(mask[0].sum().item())]
    padded, attn_mask = pad_prompt_embeddings([emb[: int(max_sequence_length)].detach()])
    return padded.to(device), attn_mask.to(device)


@torch.inference_mode()
def sample_qwen_edit_rgba(
    *, example, config, transformer, rgba_vae, scheduler, prompt_pipeline, device, weight_dtype, generator,
    enable_alpha: bool = True,
):
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift, retrieve_timesteps

    vae = rgba_vae.vae
    inference = config.get("inference", {})
    model = config["model"]
    true_cfg_scale = float(inference.get("true_cfg_scale", inference.get("guidance_scale", 4.0)))
    do_cfg = true_cfg_scale > 1.0
    num_steps = int(inference.get("num_inference_steps", 50))
    max_seq_len = int(model.get("max_sequence_length", 512))
    cond_area = int(model.get("condition_image_area", 384 * 384))

    condition_rgb = example["condition_pixel_values"].unsqueeze(0).to(device=device, dtype=weight_dtype)
    prompt_embeds, prompt_mask = _encode_prompt(
        prompt_pipeline, example["condition_pixel_values"], example["prompt"],
        device=device, max_sequence_length=max_seq_len, condition_image_area=cond_area,
    )
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)

    neg_embeds = neg_mask = None
    if do_cfg:
        neg_embeds, neg_mask = _encode_prompt(
            prompt_pipeline, example["condition_pixel_values"], str(inference.get("negative_prompt", " ")),
            device=device, max_sequence_length=max_seq_len, condition_image_area=cond_area,
        )
        neg_embeds = neg_embeds.to(dtype=weight_dtype)

    condition_latents = normalized_vae_mode(vae, condition_rgb)
    packed_condition = pack_qwen_latents(condition_latents)
    packed_latents = pack_qwen_latents(torch.randn(condition_latents.shape, generator=generator, device=device, dtype=weight_dtype))
    target_tokens = int(packed_latents.shape[1])

    eval_scheduler = type(scheduler).from_config(scheduler.config)
    sigmas = np.linspace(1.0, 1.0 / float(num_steps), num_steps)
    mu = calculate_shift(
        target_tokens,
        int(eval_scheduler.config.get("base_image_seq_len", 256)),
        int(eval_scheduler.config.get("max_image_seq_len", 4096)),
        float(eval_scheduler.config.get("base_shift", 0.5)),
        float(eval_scheduler.config.get("max_shift", 1.15)),
    )
    timesteps, _ = retrieve_timesteps(eval_scheduler, num_steps, device, sigmas=sigmas, mu=mu)
    eval_scheduler.set_begin_index(0)

    b, _, t, lh, lw = condition_latents.shape
    shape = (t, lh // 2, lw // 2)
    image_shapes = [[shape, shape] for _ in range(b)]

    for timestep in timesteps:
        model_input = torch.cat([packed_latents, packed_condition], dim=1)
        t_batch = timestep.expand(b).to(packed_latents.dtype)
        with transformer.cache_context("cond"):
            pred_flow = transformer(
                hidden_states=model_input, timestep=t_batch / 1000.0, guidance=None,
                encoder_hidden_states_mask=prompt_mask, encoder_hidden_states=prompt_embeds,
                img_shapes=image_shapes, return_dict=False,
            )[0][:, :target_tokens]
        if do_cfg:
            with transformer.cache_context("uncond"):
                neg_flow = transformer(
                    hidden_states=model_input, timestep=t_batch / 1000.0, guidance=None,
                    encoder_hidden_states_mask=neg_mask, encoder_hidden_states=neg_embeds,
                    img_shapes=image_shapes, return_dict=False,
                )[0][:, :target_tokens]
            combined = neg_flow + true_cfg_scale * (pred_flow - neg_flow)
            pred_flow = combined * (torch.norm(pred_flow, dim=-1, keepdim=True) / torch.norm(combined, dim=-1, keepdim=True).clamp_min(1e-8))
        packed_latents = eval_scheduler.step(pred_flow, timestep, packed_latents, return_dict=False)[0]

    latents = unpack_qwen_latents(
        packed_latents, height=int(condition_rgb.shape[-2]), width=int(condition_rgb.shape[-1]),
        vae_scale_factor=2 ** len(vae.temperal_downsample),
    )
    out_size = example["condition_pixel_values"].shape[-2:]
    if enable_alpha:
        rgb, score = rgba_vae(latents, operation="decode_rgba")
        if rgb.shape[-2:] != out_size:
            rgb = F.interpolate(rgb, size=out_size, mode="bilinear", align_corners=False)
        if score.shape[-2:] != out_size:
            score = F.interpolate(score, size=out_size, mode="bilinear", align_corners=False)
        return rgb, ea_score_to_alpha(score)
    rgb = rgba_vae(latents, operation="decode_rgb")
    if rgb.shape[-2:] != out_size:
        rgb = F.interpolate(rgb, size=out_size, mode="bilinear", align_corners=False)
    return rgb, None
