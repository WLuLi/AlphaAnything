"""Full-image canvas preparation for standalone image-to-RGBA inference."""
from __future__ import annotations

import math

from PIL import Image
from PIL.ImageOps import exif_transpose


def i2rgba_image_size(size: tuple[int, int]) -> tuple[int, int]:
    """Return (width, height) near 1024² pixels, aligned for all I2RGBA backbones."""
    width, height = size
    scale = math.sqrt(1024 * 1024 / (width * height))
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def prepare_i2rgba_image(image: Image.Image) -> Image.Image:
    """Orient and resize the complete RGB image without cropping or padding."""
    image = exif_transpose(image).convert("RGB")
    size = i2rgba_image_size(image.size)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image
