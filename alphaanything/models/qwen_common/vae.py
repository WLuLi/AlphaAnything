from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from alphaanything.modules.alpha_decoder import BlockTriangularAlphaDecoder


class SinglePassRGBAQwenVAE(nn.Module):
    """Frozen Qwen RGB VAE paired with a one-way hierarchical alpha branch."""

    def __init__(self, vae: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.vae = vae.requires_grad_(False).eval()
        self.alpha_decoder = BlockTriangularAlphaDecoder(self.vae.decoder, config)

    @property
    def config(self) -> Any:
        return self.vae.config

    @property
    def temperal_downsample(self) -> Any:
        return self.vae.temperal_downsample

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.alpha_decoder.load_state_dict(dict(state), strict=True)

    def _denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return latents * std + mean

    def decode_rgba(self, normalized_latents: torch.Tensor, *, clamp_rgb: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self.vae.decoder
        value = self.vae.post_quant_conv(self._denormalize(normalized_latents))
        rgb = decoder.conv_in(value)
        alpha = self.alpha_decoder.alpha_init(rgb)

        mid_resnets = list(decoder.mid_block.resnets)
        mid_attentions = list(getattr(decoder.mid_block, "attentions", []) or [None] * (len(mid_resnets) - 1))
        rgb = mid_resnets[0](rgb)
        alpha = self.alpha_decoder.mid_stage.inject(alpha, rgb, 0)
        for index, (attention, resnet) in enumerate(zip(mid_attentions, mid_resnets[1:], strict=True), start=1):
            if attention is not None:
                rgb = attention(rgb)
            rgb = resnet(rgb)
            alpha = self.alpha_decoder.mid_stage.inject(alpha, rgb, index)
        alpha = self.alpha_decoder.mid_stage.finish(alpha, rgb)

        for rgb_stage, alpha_stage in zip(decoder.up_blocks, self.alpha_decoder.up_stages, strict=True):
            alpha = alpha_stage.begin(alpha)
            for index, resnet in enumerate(rgb_stage.resnets):
                rgb = resnet(rgb)
                alpha = alpha_stage.inject(alpha, rgb, index)
            alpha = alpha_stage.finish(alpha)
            for upsampler in getattr(rgb_stage, "upsamplers", None) or []:
                rgb = upsampler(rgb)
            alpha = alpha_stage.forward_upsample(alpha, tuple(int(v) for v in rgb.shape[-2:]))

        rgb = decoder.conv_out(decoder.nonlinearity(decoder.norm_out(rgb)))
        if clamp_rgb:
            rgb = rgb.clamp(-1.0, 1.0)
        score = self.alpha_decoder.ea_score_head(alpha)
        return rgb[:, :, 0], score[:, :, 0]

    def forward(self, value: torch.Tensor, *, operation: str = "decode_rgba", clamp_rgb: bool = True):
        if operation != "decode_rgba":
            raise ValueError(f"Unsupported RGBA VAE operation: {operation!r}.")
        return self.decode_rgba(value, clamp_rgb=clamp_rgb)
