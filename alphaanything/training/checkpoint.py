from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import save_file

from alphaanything.checkpoint import ALPHA_FILENAME, CONFIG_FILENAME, adapter_metadata
from alphaanything.models.flux2_klein.vae import SinglePassRGBAFlux2VAE
from alphaanything.utils.cli import write_yaml
from alphaanything.utils.lora import lora_adapter_metadata


def resolve_state_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    state_path = path if path.is_file() else path / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"No training_state.pt found under {path}.")
    return state_path


def load_training_state(
    payload: dict[str, Any],
    transformer: torch.nn.Module,
    rgba_vae: SinglePassRGBAFlux2VAE,
    adapter_name: str,
) -> None:
    state = payload["state"]
    set_peft_model_state_dict(transformer, state["transformer_lora"], adapter_name=adapter_name)
    rgba_vae.load_adapter_state_dict(state["vae_alpha_adapter"])


def save_training_checkpoint(
    *,
    accelerator: Any,
    transformer: torch.nn.Module,
    rgba_vae: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    config: dict[str, Any],
    step: int,
    destination: Path,
) -> None:
    if not accelerator.is_main_process:
        return
    destination.mkdir(parents=True, exist_ok=True)
    transformer = accelerator.unwrap_model(transformer)
    rgba_vae = accelerator.unwrap_model(rgba_vae)
    adapter_name = str(config["model"]["adapter_name"])

    lora = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(transformer, adapter_name=adapter_name).items()}
    alpha = rgba_vae.adapter_state_dict()
    metadata = adapter_metadata(config, backbone="flux2_klein", task=config["task"], step=step)

    Flux2KleinPipeline.save_lora_weights(
        save_directory=destination,
        transformer_lora_layers=lora,
        transformer_lora_adapter_metadata=lora_adapter_metadata(transformer, adapter_name),
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in alpha.items()},
        str(destination / ALPHA_FILENAME),
        metadata={"format": "pt", "backbone": "flux2_klein", "task": config["task"], "base_model": metadata["base_model"]},
    )
    (destination / CONFIG_FILENAME).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_yaml(destination / "config.resolved.yaml", config)
    accelerator.save(
        {
            "step": step,
            "state": {"transformer_lora": lora, "vae_alpha_adapter": alpha, "adapter_name": adapter_name},
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "config": config,
        },
        destination / "training_state.pt",
    )
