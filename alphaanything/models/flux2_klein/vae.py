from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from torch import nn

from alphaanything.modules.alpha_decoder import BlockTriangularAlphaDecoder, _checkpointed_call


class SinglePassRGBAFlux2VAE(nn.Module):
    """Frozen FLUX.2 RGB VAE paired with a one-way hierarchical alpha branch."""

    def __init__(self, vae: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.vae = vae.requires_grad_(False).eval()
        self.alpha_decoder = BlockTriangularAlphaDecoder(vae.decoder, config)

    def train(self, mode: bool = True) -> SinglePassRGBAFlux2VAE:
        super().train(mode)
        self.vae.eval()
        return self

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.alpha_decoder.state_dict().items()}

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.alpha_decoder.load_state_dict(dict(state), strict=True)

    def _bn_stats(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(value)
        std = (self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).sqrt().to(value)
        return mean, std

    @torch.no_grad()
    def encode_normalized_mode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(next(self.vae.parameters()))
        raw = self.vae.encode(pixels).latent_dist.mode()
        value = Flux2KleinPipeline._patchify_latents(raw)
        mean, std = self._bn_stats(value)
        return (value - mean) / std

    def _decode_rgba(
        self, normalized_latents: torch.Tensor, *, clamp_rgb: bool = True, use_activation_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        run = partial(_checkpointed_call, use_activation_checkpoint=use_activation_checkpoint)
        value = normalized_latents.to(next(self.vae.parameters()))
        mean, std = self._bn_stats(value)
        value = Flux2KleinPipeline._unpatchify_latents(value * std + mean)
        if self.vae.post_quant_conv is not None:
            value = self.vae.post_quant_conv(value)
        decoder = self.vae.decoder
        rgb = decoder.conv_in(value)
        alpha = run(self.alpha_decoder.alpha_init, rgb.unsqueeze(2))

        resnets = list(decoder.mid_block.resnets)
        attentions = list(decoder.mid_block.attentions or [None] * (len(resnets) - 1))
        rgb = run(resnets[0], rgb, temb=None)
        alpha = self.alpha_decoder.mid_stage.inject(alpha, rgb.unsqueeze(2), 0, use_activation_checkpoint=use_activation_checkpoint)
        for index, (attention, resnet) in enumerate(zip(attentions, resnets[1:], strict=True), start=1):
            if attention is not None:
                rgb = run(attention, rgb)
            rgb = run(resnet, rgb, temb=None)
            alpha = self.alpha_decoder.mid_stage.inject(alpha, rgb.unsqueeze(2), index, use_activation_checkpoint=use_activation_checkpoint)
        alpha = self.alpha_decoder.mid_stage.finish(alpha, rgb.unsqueeze(2), use_activation_checkpoint=use_activation_checkpoint)

        for rgb_stage, alpha_stage in zip(decoder.up_blocks, self.alpha_decoder.up_stages, strict=True):
            alpha = alpha_stage.begin(alpha)
            for index, resnet in enumerate(rgb_stage.resnets):
                rgb = run(resnet, rgb, temb=None)
                alpha = alpha_stage.inject(alpha, rgb.unsqueeze(2), index, use_activation_checkpoint=use_activation_checkpoint)
            alpha = alpha_stage.finish(alpha, use_activation_checkpoint=use_activation_checkpoint)
            for upsampler in rgb_stage.upsamplers or []:
                rgb = run(upsampler, rgb)
            alpha = alpha_stage.forward_upsample(alpha, tuple(rgb.shape[-2:]), use_activation_checkpoint=use_activation_checkpoint)

        rgb = decoder.conv_out(decoder.conv_act(decoder.conv_norm_out(rgb)))
        score = self.alpha_decoder.ea_score_head(alpha, use_activation_checkpoint=use_activation_checkpoint)[:, :, 0]
        return rgb.clamp(-1.0, 1.0) if clamp_rgb else rgb, score

    def decode_rgb(self, normalized_latents: torch.Tensor, *, clamp_rgb: bool = True) -> torch.Tensor:
        value = normalized_latents.to(next(self.vae.parameters()))
        mean, std = self._bn_stats(value)
        value = Flux2KleinPipeline._unpatchify_latents(value * std + mean)
        rgb = self.vae.decode(value, return_dict=False)[0]
        return rgb.clamp(-1.0, 1.0) if clamp_rgb else rgb

    def forward(
        self,
        value: torch.Tensor,
        *,
        operation: str,
        use_activation_checkpoint: bool = False,
        clamp_rgb: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if operation == "encode_normalized_mode":
            return self.encode_normalized_mode(value)
        if operation == "decode_rgb":
            return self.decode_rgb(value, clamp_rgb=clamp_rgb)
        if operation == "decode_rgba":
            ckpt = use_activation_checkpoint and torch.is_grad_enabled() and (
                value.requires_grad or any(p.requires_grad for p in self.alpha_decoder.parameters())
            )
            return self._decode_rgba(value, clamp_rgb=clamp_rgb, use_activation_checkpoint=ckpt)
        raise ValueError(f"Unknown RGBA VAE operation: {operation!r}.")
