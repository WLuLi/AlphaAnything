from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _opencv_area_resize_axis(value: torch.Tensor, output_size: int, axis: int) -> torch.Tensor:
    """Downsample one spatial axis with OpenCV INTER_AREA overlap weights."""
    input_size = int(value.shape[axis])
    if output_size == input_size:
        return value

    scale = input_size / output_size
    contributor_count = math.ceil(scale) + 1
    out_idx = torch.arange(output_size, device=value.device, dtype=torch.float64)
    starts = out_idx * scale
    ends = (out_idx + 1.0) * scale
    first = torch.floor(starts).to(torch.long)
    offsets = torch.arange(contributor_count, device=value.device, dtype=torch.long)
    src_idx = first[:, None] + offsets[None, :]
    valid = src_idx < input_size
    src_idx = src_idx.clamp(max=input_size - 1)
    src_f = src_idx.to(torch.float64)
    weights = ((torch.minimum(ends[:, None], src_f + 1.0) - torch.maximum(starts[:, None], src_f)).clamp_min(0.0) / scale) * valid

    moved = value.movedim(axis, -1)
    gathered = moved.index_select(-1, src_idx.flatten()).reshape(*moved.shape[:-1], output_size, contributor_count)
    return (gathered * weights.to(dtype=value.dtype)).sum(dim=-1).movedim(-1, axis)


def _opencv_area_resize_2d(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    output_height, output_width = size
    return _opencv_area_resize_axis(_opencv_area_resize_axis(value, output_width, axis=-1), output_height, axis=-2)


def layerdiffuse_pad_rgb(rgb01: torch.Tensor, alpha01: torch.Tensor, downsample_factor: float = 1.2) -> torch.Tensor:
    """Multiscale Gaussian-equivalent RGB color padding for hidden transparent regions (LayerDiffuse)."""
    out_dtype = rgb01.dtype
    color = rgb01.float().clamp(0.0, 1.0)
    alpha = alpha01.float().clamp(0.0, 1.0)
    premul = color * alpha
    pyramid: list[tuple[torch.Tensor, torch.Tensor]] = [(premul, alpha)]

    while min(premul.shape[-2:]) > 1:
        h, w = premul.shape[-2:]
        next_size = (max(1, int(h / downsample_factor)), max(1, int(w / downsample_factor)))
        resized = _opencv_area_resize_2d(torch.cat((premul, alpha), dim=1), next_size)
        premul, alpha = resized[:, :3], resized[:, 3:4]
        pyramid.append((premul, alpha))

    coarse_color, coarse_alpha = pyramid[-1]
    fg = coarse_color.sum(dim=(-2, -1), keepdim=True) / coarse_alpha.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    for layer_color, layer_alpha in reversed(pyramid):
        fg = F.interpolate(fg, size=layer_color.shape[-2:], mode="bilinear", align_corners=False)
        fg = layer_color + fg * (1.0 - layer_alpha)
    return fg.clamp(0.0, 1.0).to(dtype=out_dtype)


def resize_straight_rgba_premultiplied(rgba01: torch.Tensor, size: tuple[int, int] | list[int]) -> torch.Tensor:
    """Resize straight RGBA in [0, 1] through premultiplied RGB to prevent zero-alpha color bleeding."""
    squeeze = rgba01.ndim == 3
    rgba = rgba01.unsqueeze(0) if squeeze else rgba01
    out_dtype = rgba.dtype
    rgba = rgba.float().clamp(0.0, 1.0)
    alpha = rgba[:, 3:4]
    premul = rgba[:, :3] * alpha
    target_size = (int(size[0]), int(size[1]))

    alpha = F.interpolate(alpha, size=target_size, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
    premul = F.interpolate(premul, size=target_size, mode="bicubic", align_corners=False, antialias=True).clamp_min(0.0)
    premul = torch.minimum(premul, alpha)

    eps = torch.finfo(premul.dtype).eps
    rgb = torch.where(alpha > 0.0, premul / alpha.clamp_min(eps), torch.zeros_like(premul)).clamp(0.0, 1.0)
    out = torch.cat([rgb, alpha], dim=1).to(dtype=out_dtype)
    return out[0] if squeeze else out


def rgba_to_layerdiffuse_rgb(rgba: torch.Tensor) -> torch.Tensor:
    """Convert normalized RGBA [-1, 1] to LayerDiffuse-padded RGB [-1, 1]."""
    rgba01 = ((rgba.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    padded = layerdiffuse_pad_rgb(rgba01[:, :3], rgba01[:, 3:4])
    return (padded * 2.0 - 1.0).to(dtype=rgba.dtype)
