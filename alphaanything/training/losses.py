from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from alphaanything.modules.alpha_math import (
    ea_loss_per_sample,
    ea_score_to_alpha,
    sigma_image_loss_weight,
)


def flow_ea_loss(
    flow_per_sample: torch.Tensor,
    score: torch.Tensor,
    target_alpha: torch.Tensor,
    sigmas: torch.Tensor,
    loss_config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Combined latent flow-matching loss and sigma-weighted Endpoint-Aware Alpha Loss."""
    flow_weight = float(loss_config.get("latent_flow_weight", 1.0))
    ea_weight = float(loss_config.get("ea_loss_weight", 1.0))
    root, transparent, opaque = ea_loss_per_sample(score, target_alpha)
    timestep_weight = sigma_image_loss_weight(
        sigmas,
        reference_sigma=float(loss_config.get("image_weight_reference_sigma", 0.5)),
        epsilon=float(loss_config.get("image_weight_epsilon", 1e-4)),
    )
    flow = flow_weight * flow_per_sample.float()
    alpha = ea_weight * timestep_weight * root
    return {
        "loss": (flow + alpha).mean(),
        "latent_flow_loss": flow_per_sample.float().mean(),
        "ea_loss": root.mean(),
        "ea_transparent_root": transparent.mean(),
        "ea_opaque_root": opaque.mean(),
        "ea_loss_weighted": alpha.mean(),
        "alpha_mean": ea_score_to_alpha(score.detach()).mean(),
    }
