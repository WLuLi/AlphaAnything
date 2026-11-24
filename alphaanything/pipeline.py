"""High-level AlphaAnything inference pipeline with hot-swappable RGB/RGBA and multi-adapter support."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from alphaanything.checkpoint import EXTRACTION_PREFIX, load_adapter_bundle, read_adapter_config, set_lora_enabled
from alphaanything.data.canvas import prepare_i2rgba_image
from alphaanything.utils.image import tensor_to_rgb_pil, tensor_to_rgba_pil
from alphaanything.utils.model import resolve_weight_dtype


@dataclass
class _AdapterSlot:
    name: str
    task: str
    config: dict[str, Any]
    scale: float
    rgba_vae: torch.nn.Module


class AlphaAnythingPipeline:
    """Single-deployment pipeline supporting runtime RGB/RGBA toggling and multiple adapters."""

    def __init__(
        self,
        *,
        backbone: str,
        base_model: str,
        backend: Any,
        pipe: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.backbone = backbone
        self.base_model = base_model
        self._backend = backend
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        self._adapters: dict[str, _AdapterSlot] = {}
        self._active_adapter: str | None = None
        self._alpha_enabled: bool = True

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        base_model: str | Path | None = None,
        adapter_name: str | None = None,
        device: str | torch.device = "cuda",
        mixed_precision: str | None = None,
        local_files_only: bool = False,
        enable_alpha: bool = True,
    ) -> AlphaAnythingPipeline:
        config = read_adapter_config(checkpoint)
        if base_model is not None:
            config["base_model"] = str(base_model)
            config["model"]["pretrained_model_name_or_path"] = str(base_model)
        config["model"]["local_files_only"] = bool(local_files_only)

        resolved_device = torch.device(device)
        precision = mixed_precision if mixed_precision is not None else config["inference"]["mixed_precision"]
        dtype = (
            torch.float32
            if (resolved_device.type == "cpu" and mixed_precision is None)
            else resolve_weight_dtype(precision)
        )
        backbone = str(config["backbone"])
        backend = importlib.import_module(f"alphaanything.models.{backbone}.pipeline")
        pipe = backend.load_base_pipeline(config, resolved_device, dtype)

        instance = cls(
            backbone=backbone,
            base_model=str(config["base_model"]),
            backend=backend,
            pipe=pipe,
            device=resolved_device,
            dtype=dtype,
        )
        instance._alpha_enabled = bool(enable_alpha)
        instance.load_adapter(checkpoint, name=adapter_name, activate=True)
        return instance

    @property
    def adapters(self) -> list[str]:
        return list(self._adapters.keys())

    @property
    def active_adapter(self) -> str:
        if self._active_adapter is None:
            raise RuntimeError("No adapter is loaded.")
        return self._active_adapter

    @property
    def active_config(self) -> dict[str, Any]:
        return self._adapters[self.active_adapter].config

    @property
    def alpha_enabled(self) -> bool:
        return self._alpha_enabled

    def _sync_lora_state(self) -> None:
        slot = self._adapters[self.active_adapter]
        set_lora_enabled(self.pipe, self._alpha_enabled, adapter_name=slot.name, scale=slot.scale)

    def load_adapter(
        self,
        checkpoint: str | Path,
        *,
        name: str | None = None,
        activate: bool = False,
    ) -> str:
        config = read_adapter_config(checkpoint)
        if str(config["backbone"]) != self.backbone:
            raise ValueError(
                f"Adapter backbone {config['backbone']!r} does not match loaded pipeline backbone {self.backbone!r}."
            )
        config["base_model"] = self.base_model
        config["model"]["pretrained_model_name_or_path"] = self.base_model

        resolved_name = str(name or config["lora"].get("adapter_name", "alphaanything"))
        if resolved_name in self._adapters:
            raise ValueError(f"Adapter {resolved_name!r} is already loaded.")

        rgba_vae = self._backend.create_rgba_vae(self.pipe, config["vae_alpha_adapter"], self.device, self.dtype)
        resolved_name, scale = load_adapter_bundle(
            self.pipe,
            rgba_vae,
            checkpoint,
            config,
            adapter_name=resolved_name,
            activate=False,
        )
        self.pipe.transformer.to(device=self.device)
        rgba_vae.to(device=self.device, dtype=self.dtype).eval()

        self._adapters[resolved_name] = _AdapterSlot(
            name=resolved_name,
            task=str(config["task"]),
            config=config,
            scale=scale,
            rgba_vae=rgba_vae,
        )
        if activate or self._active_adapter is None:
            self._active_adapter = resolved_name
        self._sync_lora_state()
        return resolved_name

    def set_adapter(self, name: str) -> None:
        if name not in self._adapters:
            raise KeyError(f"Unknown adapter {name!r}. Loaded adapters: {self.adapters}.")
        self._active_adapter = name
        self._sync_lora_state()

    def enable_alpha(self, adapter: str | None = None) -> None:
        if adapter is not None:
            if adapter not in self._adapters:
                raise KeyError(f"Unknown adapter {adapter!r}. Loaded adapters: {self.adapters}.")
            self._active_adapter = adapter
        self._alpha_enabled = True
        self._sync_lora_state()

    def disable_alpha(self) -> None:
        self._alpha_enabled = False
        self._sync_lora_state()

    @contextmanager
    def alpha_mode(self, enabled: bool, *, adapter: str | None = None):
        prev_enabled = self._alpha_enabled
        prev_adapter = self._active_adapter
        try:
            if enabled:
                self.enable_alpha(adapter=adapter)
            else:
                if adapter is not None:
                    self.set_adapter(adapter)
                self.disable_alpha()
            yield self
        finally:
            self._active_adapter = prev_adapter
            self._alpha_enabled = prev_enabled
            if self._active_adapter is not None:
                self._sync_lora_state()

    @staticmethod
    def _resolve_prompt(prompt: str | None, description: str | None, *, allow_description: bool) -> str:
        if prompt and description:
            raise ValueError("Pass either prompt or description, not both.")
        if description:
            if not allow_description:
                raise ValueError("description is only supported for image-conditioned tasks; pass prompt instead.")
            prompt = f"{EXTRACTION_PREFIX} {str(description).strip()}"
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Missing prompt.")
        return prompt

    @staticmethod
    def _prepare_image(
        image: Image.Image | str | Path | None,
        *,
        height: int | None,
        width: int | None,
    ) -> tuple[Image.Image | None, int | None, int | None]:
        if image is None:
            return None, height, width
        if isinstance(image, (str, Path)):
            with Image.open(image) as src:
                pil_image = src.convert("RGB")
        else:
            pil_image = image.convert("RGB")
        if height is None and width is None:
            pil_image = prepare_i2rgba_image(pil_image)
            return pil_image, pil_image.height, pil_image.width
        if height is not None and width is not None:
            pil_image = pil_image.resize((int(width), int(height)), resample=Image.Resampling.LANCZOS)
            return pil_image, int(height), int(width)
        raise ValueError("Specify both height and width, or omit both to preserve the input aspect ratio.")

    @torch.inference_mode()
    def generate_tensors(
        self,
        *,
        prompt: str | None = None,
        description: str | None = None,
        image: Image.Image | str | Path | None = None,
        enable_alpha: bool | None = None,
        adapter: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target_enabled = self._alpha_enabled if enable_alpha is None else bool(enable_alpha)
        target_adapter = self.active_adapter if adapter is None else adapter
        if target_adapter not in self._adapters:
            raise KeyError(f"Unknown adapter {target_adapter!r}. Loaded adapters: {self.adapters}.")

        slot = self._adapters[target_adapter]
        if target_enabled:
            if slot.task == "image_to_rgba" and image is None:
                raise ValueError(f"Adapter {slot.name!r} ({slot.task}) requires an input image.")
            if slot.task == "text_to_rgba" and image is not None:
                raise ValueError(f"Adapter {slot.name!r} ({slot.task}) does not accept an input image.")
        else:
            if self.backbone == "qwen_image_edit" and image is None:
                raise ValueError("qwen_image_edit requires an input image even when alpha is disabled.")
            if self.backbone == "qwen_image" and image is not None:
                raise ValueError("qwen_image does not accept an input image.")

        allow_description = slot.task == "image_to_rgba" or image is not None
        resolved_prompt = self._resolve_prompt(prompt, description, allow_description=allow_description)
        prepared_image, resolved_height, resolved_width = self._prepare_image(image, height=height, width=width)

        params = slot.config["inference"]
        resolved_height = int(params["height"] if resolved_height is None else resolved_height)
        resolved_width = int(params["width"] if resolved_width is None else resolved_width)
        steps = int(params["num_inference_steps"] if num_inference_steps is None else num_inference_steps)
        guidance = float(params["guidance_scale"] if guidance_scale is None else guidance_scale)
        neg_prompt = str(params["negative_prompt"] if negative_prompt is None else negative_prompt)
        if generator is None:
            resolved_seed = int(params["seed"] if seed is None else seed)
            generator = torch.Generator(device=self.device).manual_seed(resolved_seed)

        with self.alpha_mode(target_enabled, adapter=target_adapter):
            return self._backend.generate(
                self.pipe,
                slot.rgba_vae,
                prompt=resolved_prompt,
                image=prepared_image,
                height=resolved_height,
                width=resolved_width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=neg_prompt,
                generator=generator,
                config=slot.config,
                enable_alpha=target_enabled,
            )

    def __call__(
        self,
        prompt: str | None = None,
        *,
        description: str | None = None,
        image: Image.Image | str | Path | None = None,
        enable_alpha: bool | None = None,
        adapter: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Image.Image:
        rgb, alpha = self.generate_tensors(
            prompt=prompt,
            description=description,
            image=image,
            enable_alpha=enable_alpha,
            adapter=adapter,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            generator=generator,
        )
        if alpha is None:
            return tensor_to_rgb_pil(rgb)
        return tensor_to_rgba_pil(rgb, alpha)
