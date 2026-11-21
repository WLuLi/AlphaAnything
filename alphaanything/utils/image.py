from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def to01(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.float() + 1.0) * 0.5).clamp(0, 1)


def rgba_on_checker(rgb: torch.Tensor, alpha: torch.Tensor, squares: int = 16) -> torch.Tensor:
    batch_size, _, height, width = rgb.shape
    y = torch.arange(height, device=rgb.device)[:, None]
    x = torch.arange(width, device=rgb.device)[None, :]
    checker = (((x // squares + y // squares) % 2).float() * 0.18 + 0.72).view(1, 1, height, width)
    checker = checker.expand(batch_size, 3, height, width)
    rgb01 = to01(rgb)
    alpha = alpha.float().clamp(0, 1)
    return rgb01 * alpha + checker * (1.0 - alpha)


def _to_uint8_hwc(image: torch.Tensor):
    return (image.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype("uint8")


@torch.no_grad()
def save_infer_sample(
    output_dir: Path,
    prefix: str,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    *,
    paths: dict[str, Path] | None = None,
) -> dict[str, str]:
    rgb01 = to01(rgb.float().cpu())
    alpha01 = alpha.float().cpu().clamp(0, 1)
    checker = rgba_on_checker(rgb.unsqueeze(0).float().cpu(), alpha.unsqueeze(0).float().cpu())[0]
    rgba = torch.cat([rgb01, alpha01], dim=0)

    paths = paths if paths is not None else {
        "rgba": output_dir / f"{prefix}_rgba.png",
        "rgb": output_dir / f"{prefix}_rgb.png",
        "alpha": output_dir / f"{prefix}_alpha.png",
        "checker": output_dir / f"{prefix}_checker.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_hwc(rgba), mode="RGBA").save(paths["rgba"])
    Image.fromarray(_to_uint8_hwc(rgb01), mode="RGB").save(paths["rgb"])
    Image.fromarray(_to_uint8_hwc(alpha01)[..., 0], mode="L").save(paths["alpha"])
    Image.fromarray(_to_uint8_hwc(checker), mode="RGB").save(paths["checker"])
    return {key: str(path) for key, path in paths.items()}


def save_normalized_rgb(path: Path, image: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8_hwc(to01(image.detach())), mode="RGB").save(path)
    return str(path)
