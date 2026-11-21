"""Shared latent and prompt tensor utilities for Qwen backbones."""
from __future__ import annotations

import torch


def normalized_vae_mode(vae, pixels: torch.Tensor) -> torch.Tensor:
    encoded = vae.encode(pixels.unsqueeze(2)).latent_dist.mode()
    mean = torch.as_tensor(vae.config.latents_mean, device=encoded.device, dtype=encoded.dtype).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=encoded.device, dtype=encoded.dtype).view(1, -1, 1, 1, 1)
    return (encoded - mean) / std


def pack_qwen_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, _, h, w = latents.shape
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_qwen_latents(packed: torch.Tensor, *, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    b, _, c4 = packed.shape
    lh = 2 * (int(height) // (int(vae_scale_factor) * 2))
    lw = 2 * (int(width) // (int(vae_scale_factor) * 2))
    return packed.view(b, lh // 2, lw // 2, c4 // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c4 // 4, 1, lh, lw)


def pad_prompt_embeddings(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    feat_dim = int(embeddings[0].shape[-1])
    max_len = max(int(e.shape[0]) for e in embeddings)
    out = embeddings[0].new_zeros((len(embeddings), max_len, feat_dim))
    mask = torch.zeros((len(embeddings), max_len), dtype=torch.long)
    for i, emb in enumerate(embeddings):
        n = int(emb.shape[0])
        out[i, :n] = emb
        mask[i, :n] = 1
    return out, mask
