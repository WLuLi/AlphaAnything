from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def _checkpointed_call(function, *args, use_activation_checkpoint: bool = False, **kwargs):
    if use_activation_checkpoint and torch.is_grad_enabled():
        return activation_checkpoint(function, *args, use_reentrant=False, **kwargs)
    return function(*args, **kwargs)


def _group_count(channels: int, maximum: int = 32) -> int:
    for groups in range(min(int(channels), int(maximum)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _spatial_resize_nearest_5d(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize spatial H/W of BCTHW features."""
    batch, channels, frames, height, width = value.shape
    if (height, width) == tuple(size):
        return value
    flat = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flat = F.interpolate(flat, size=size, mode="nearest-exact")
    return flat.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)


def _conv_out_channels(module: nn.Module) -> int:
    out_dim = getattr(module, "out_dim", None)
    if out_dim is not None:
        return int(out_dim)
    for name in ("conv2", "conv", "conv_shortcut"):
        conv = getattr(module, name, None)
        if getattr(conv, "out_channels", None) is not None:
            return int(conv.out_channels)
    raise TypeError(f"Cannot infer output channels for decoder block {type(module)!r}.")


class RGBToAlphaLoCon(nn.Module):
    """Low-rank spatial projection used on one-way RGB -> alpha feature reads."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        *,
        alpha: float,
        dropout: float,
        output_init_std: float,
    ) -> None:
        super().__init__()
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Conv3d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = nn.Conv3d(rank, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.normal_(self.lora_B.weight, mean=0.0, std=float(output_init_std))

    def forward(self, rgb_feature: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(rgb_feature))) * self.scaling


class AlphaResBlock(nn.Module):
    """Spatial ResBlock for the alpha branch with zero-initialized residual output."""

    def __init__(self, channels: int, *, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm2 = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(value)))
        residual = self.conv2(self.dropout(F.silu(self.norm2(residual))))
        return value + residual


class AlphaInit(nn.Module):
    """Initialize the alpha branch state from the RGB decoder's conv_in feature."""

    def __init__(self, rgb_channels: int, alpha_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(rgb_channels, alpha_channels, kernel_size=1)
        self.norm = nn.GroupNorm(_group_count(alpha_channels), alpha_channels)
        self.spatial = nn.Conv3d(alpha_channels, alpha_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)
        nn.init.kaiming_normal_(self.spatial.weight, nonlinearity="relu")
        nn.init.zeros_(self.spatial.bias)

    def forward(self, rgb_feature: torch.Tensor) -> torch.Tensor:
        return self.spatial(F.silu(self.norm(self.proj(rgb_feature))))


class RGBToAlphaCrossAttention(nn.Module):
    """One-way bottleneck cross-attention: alpha queries frozen RGB keys/values."""

    def __init__(self, alpha_channels: int, rgb_channels: int, *, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(alpha_channels) // self.heads
        self.alpha_norm = nn.GroupNorm(_group_count(alpha_channels), alpha_channels)
        self.rgb_norm = nn.GroupNorm(_group_count(rgb_channels), rgb_channels)
        self.to_q = nn.Conv2d(alpha_channels, alpha_channels, kernel_size=1)
        self.to_k = nn.Conv2d(rgb_channels, alpha_channels, kernel_size=1)
        self.to_v = nn.Conv2d(rgb_channels, alpha_channels, kernel_size=1)
        self.to_out = nn.Conv2d(alpha_channels, alpha_channels, kernel_size=1)
        for proj in (self.to_q, self.to_k, self.to_v):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _tokens(self, value: torch.Tensor) -> torch.Tensor:
        batch_frames, _, height, width = value.shape
        value = value.flatten(2).transpose(1, 2).contiguous()
        return value.reshape(batch_frames, height * width, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        batch, alpha_channels, frames, height, width = alpha.shape
        rgb_channels = rgb.shape[1]
        alpha_2d = alpha.permute(0, 2, 1, 3, 4).reshape(batch * frames, alpha_channels, height, width)
        rgb_2d = rgb.permute(0, 2, 1, 3, 4).reshape(batch * frames, rgb_channels, height, width)
        query = self._tokens(self.to_q(self.alpha_norm(alpha_2d)))
        norm_rgb = self.rgb_norm(rgb_2d)
        key = self._tokens(self.to_k(norm_rgb))
        value = self._tokens(self.to_v(norm_rgb))
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch * frames, height * width, alpha_channels)
        attended = self.to_out(attended.transpose(1, 2).reshape(batch * frames, alpha_channels, height, width))
        return alpha + attended.reshape(batch, frames, alpha_channels, height, width).permute(0, 2, 1, 3, 4)


class LearnedAlphaUpsample(nn.Module):
    """Nearest spatial resize followed by a 3x3 spatial convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return self.conv(_spatial_resize_nearest_5d(value, size))


class AlphaDecoderStage(nn.Module):
    """Upsampling stage of the alpha branch reading RGB ResBlock features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rgb_feature_channels: Sequence[int],
        *,
        injection_rank: int,
        injection_alpha: float,
        injection_init_std: float,
        dropout: float,
        res_blocks: int,
        learned_upsample: bool,
        injection_gate_init: float,
    ) -> None:
        super().__init__()
        if int(in_channels) == int(out_channels):
            self.transition: nn.Module = nn.Identity()
        else:
            self.transition = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=(1, 3, 3), padding=(0, 1, 1))
            nn.init.kaiming_normal_(self.transition.weight, nonlinearity="relu")
            nn.init.zeros_(self.transition.bias)
        self.injections = nn.ModuleList([
            RGBToAlphaLoCon(
                int(rgb_c), int(out_channels), int(injection_rank),
                alpha=float(injection_alpha), dropout=float(dropout), output_init_std=float(injection_init_std),
            )
            for rgb_c in rgb_feature_channels
        ])
        self.injection_normalization = 1.0 / math.sqrt(len(self.injections))
        self.injection_gate = nn.Parameter(torch.tensor(float(injection_gate_init), dtype=torch.float32))
        self.blocks = nn.ModuleList([AlphaResBlock(int(out_channels), dropout=float(dropout)) for _ in range(res_blocks)])
        self.upsample = LearnedAlphaUpsample(int(out_channels)) if learned_upsample else None

    def begin(self, alpha: torch.Tensor) -> torch.Tensor:
        return self.transition(alpha)

    def inject(
        self, alpha: torch.Tensor, rgb_feature: torch.Tensor, index: int, *, use_activation_checkpoint: bool = False,
    ) -> torch.Tensor:
        contrib = _checkpointed_call(self.injections[index], rgb_feature, use_activation_checkpoint=use_activation_checkpoint)
        scale = self.injection_gate.to(dtype=contrib.dtype) * self.injection_normalization
        return alpha + contrib * scale

    def finish(self, alpha: torch.Tensor, *, use_activation_checkpoint: bool = False) -> torch.Tensor:
        for block in self.blocks:
            alpha = _checkpointed_call(block, alpha, use_activation_checkpoint=use_activation_checkpoint)
        return alpha

    def forward_upsample(
        self, alpha: torch.Tensor, size: tuple[int, int], *, use_activation_checkpoint: bool = False,
    ) -> torch.Tensor:
        if self.upsample is None:
            return alpha
        return _checkpointed_call(self.upsample, alpha, size, use_activation_checkpoint=use_activation_checkpoint)


class AlphaMidStage(nn.Module):
    """Bottleneck alpha stage with one RGB -> alpha cross-attention block."""

    def __init__(
        self,
        channels: int,
        rgb_feature_channels: Sequence[int],
        *,
        injection_rank: int,
        injection_alpha: float,
        injection_init_std: float,
        dropout: float,
        res_blocks: int,
        attention_heads: int,
        injection_gate_init: float,
    ) -> None:
        super().__init__()
        self.injections = nn.ModuleList([
            RGBToAlphaLoCon(
                int(rgb_c), int(channels), int(injection_rank),
                alpha=float(injection_alpha), dropout=float(dropout), output_init_std=float(injection_init_std),
            )
            for rgb_c in rgb_feature_channels
        ])
        self.injection_normalization = 1.0 / math.sqrt(len(self.injections))
        self.injection_gate = nn.Parameter(torch.tensor(float(injection_gate_init), dtype=torch.float32))
        self.blocks = nn.ModuleList([AlphaResBlock(int(channels), dropout=float(dropout)) for _ in range(res_blocks)])
        self.cross_attention = RGBToAlphaCrossAttention(
            int(channels), int(rgb_feature_channels[-1]), heads=int(attention_heads),
        )

    def inject(
        self, alpha: torch.Tensor, rgb_feature: torch.Tensor, index: int, *, use_activation_checkpoint: bool = False,
    ) -> torch.Tensor:
        contrib = _checkpointed_call(self.injections[index], rgb_feature, use_activation_checkpoint=use_activation_checkpoint)
        scale = self.injection_gate.to(dtype=contrib.dtype) * self.injection_normalization
        return alpha + contrib * scale

    def finish(
        self, alpha: torch.Tensor, final_rgb_feature: torch.Tensor, *, use_activation_checkpoint: bool = False,
    ) -> torch.Tensor:
        alpha = _checkpointed_call(self.blocks[0], alpha, use_activation_checkpoint=use_activation_checkpoint)
        alpha = _checkpointed_call(
            self.cross_attention, alpha, final_rgb_feature, use_activation_checkpoint=use_activation_checkpoint,
        )
        for block in self.blocks[1:]:
            alpha = _checkpointed_call(block, alpha, use_activation_checkpoint=use_activation_checkpoint)
        return alpha


class EAScoreHead(nn.Module):
    """Alpha head predicting the raw score for the Endpoint-Aware alpha mapping."""

    def __init__(
        self, in_channels: int, hidden_channels: int, *, dropout: float, output_init_std: float,
    ) -> None:
        super().__init__()
        self.input = nn.Conv3d(int(in_channels), int(hidden_channels), kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.blocks = nn.ModuleList([AlphaResBlock(int(hidden_channels), dropout=float(dropout)) for _ in range(2)])
        self.output = nn.Conv3d(int(hidden_channels), 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu")
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.output.bias)

    def forward(self, alpha: torch.Tensor, *, use_activation_checkpoint: bool = False) -> torch.Tensor:
        alpha = self.input(alpha)
        for block in self.blocks:
            alpha = _checkpointed_call(block, alpha, use_activation_checkpoint=use_activation_checkpoint)
        return self.output(alpha)


class BlockTriangularAlphaDecoder(nn.Module):
    """Hierarchical alpha branch with strictly one-way RGB feature sharing."""

    def __init__(self, decoder: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__()
        mid_rgb_channels = [_conv_out_channels(block) for block in decoder.mid_block.resnets]
        up_rgb_channels = [[_conv_out_channels(block) for block in stage.resnets] for stage in decoder.up_blocks]
        stage_channels = [int(v) for v in config.get("stage_channels", [256, 256, 192, 128, 96])]
        stage_gates = [float(v) for v in config.get("injection_stage_gates", [1.0, 0.75, 0.5, 0.25, 0.25])]
        injection_rank = int(config.get("injection_rank", 64))
        injection_alpha = float(config.get("injection_alpha", injection_rank))
        injection_init_std = float(config.get("injection_init_std", 1.0e-4))
        dropout = float(config.get("dropout", 0.0))
        res_blocks = int(config.get("res_blocks_per_stage", 2))
        attention_heads = int(config.get("attention_heads", 8))
        head_channels = int(config.get("head_channels", 128))
        score_init_std = float(config.get("score_output_init_std", 1.0e-4))

        self.alpha_init = AlphaInit(int(decoder.conv_in.out_channels), stage_channels[0])
        self.mid_stage = AlphaMidStage(
            stage_channels[0],
            mid_rgb_channels,
            injection_rank=injection_rank,
            injection_alpha=injection_alpha,
            injection_init_std=injection_init_std,
            dropout=dropout,
            res_blocks=res_blocks,
            attention_heads=attention_heads,
            injection_gate_init=stage_gates[0],
        )
        self.up_stages = nn.ModuleList()
        prev_channels = stage_channels[0]
        for up_block, out_channels, rgb_channels, gate_init in zip(
            decoder.up_blocks, stage_channels[1:], up_rgb_channels, stage_gates[1:], strict=True,
        ):
            has_upsample = bool(getattr(up_block, "upsamplers", None))
            self.up_stages.append(
                AlphaDecoderStage(
                    prev_channels,
                    out_channels,
                    rgb_channels,
                    injection_rank=injection_rank,
                    injection_alpha=injection_alpha,
                    injection_init_std=injection_init_std,
                    dropout=dropout,
                    res_blocks=res_blocks,
                    learned_upsample=has_upsample,
                    injection_gate_init=gate_init,
                )
            )
            prev_channels = out_channels
        self.ea_score_head = EAScoreHead(
            stage_channels[-1], head_channels, dropout=dropout, output_init_std=score_init_std,
        )
