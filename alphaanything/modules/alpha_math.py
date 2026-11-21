from __future__ import annotations

import math

import torch


def _mean_per_sample(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1).mean(dim=1)


def _zero_safe_sqrt(value: torch.Tensor) -> torch.Tensor:
    """Square root with zero subgradient at exact zero."""
    tiny = torch.finfo(value.dtype).tiny
    return torch.where(value > 0.0, torch.sqrt(value.clamp_min(tiny)), torch.zeros_like(value))


def alpha_to_ea_coordinate(alpha: torch.Tensor) -> torch.Tensor:
    """Map alpha in [0, 1] to its Endpoint-Aware coordinate in [0, 1]."""
    alpha = alpha.float().clamp(0.0, 1.0)
    return torch.atan2(torch.sqrt(alpha), torch.sqrt(1.0 - alpha)) * (2.0 / math.pi)


def ea_coordinate_from_score(score: torch.Tensor) -> torch.Tensor:
    """Map raw alpha-head score to the Endpoint-Aware coordinate in [0, 1]."""
    return ((score.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def ea_coordinate_to_alpha(coordinate: torch.Tensor) -> torch.Tensor:
    """Map Endpoint-Aware coordinate to alpha via sin^2(pi * u / 2)."""
    return torch.sin(coordinate.float() * (math.pi / 2.0)).square()


def ea_score_to_alpha(score: torch.Tensor) -> torch.Tensor:
    """Convert raw alpha-head score to physical alpha in [0, 1]."""
    return ea_coordinate_to_alpha(ea_coordinate_from_score(score))


def ea_pointwise_gap(
    score: torch.Tensor,
    target_coordinate: torch.Tensor,
) -> torch.Tensor:
    """Pointwise Endpoint-Aware gap with nonzero gradient outside [-1, 1]."""
    score = score.float()
    target_coordinate = target_coordinate.float().clamp(0.0, 1.0)
    predicted = ea_coordinate_from_score(score)
    active_gap = (predicted - target_coordinate).square()
    lower_gap = target_coordinate * (target_coordinate - 1.0 - score)
    upper_gap = (1.0 - target_coordinate) * (score - target_coordinate)
    return torch.where(score <= -1.0, lower_gap, torch.where(score >= 1.0, upper_gap, active_gap))


def ea_loss_per_sample(
    score: torch.Tensor,
    target_alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Endpoint-Aware Alpha Loss: returns (total, transparent_root, opaque_root) per sample."""
    score = score.float()
    target_alpha = target_alpha.float().clamp(0.0, 1.0)
    gap = ea_pointwise_gap(score, alpha_to_ea_coordinate(target_alpha))
    transparent_root = _zero_safe_sqrt(_mean_per_sample((1.0 - target_alpha) * gap))
    opaque_root = _zero_safe_sqrt(_mean_per_sample(target_alpha * gap))
    return transparent_root + opaque_root, transparent_root, opaque_root


def sigma_image_loss_weight(
    sigmas: torch.Tensor,
    *,
    reference_sigma: float = 0.5,
    epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Downweight image-space alpha loss at high noise levels."""
    sigma_per_sample = sigmas.float().reshape(sigmas.shape[0], -1)[:, 0]
    ratio = float(reference_sigma) ** 2 / (sigma_per_sample.square() + float(epsilon))
    return torch.minimum(torch.ones_like(ratio), ratio)
