from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


def resolve_weight_dtype(mixed_precision: str | None) -> torch.dtype:
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


@contextmanager
def isolated_scheduler(pipe):
    """Run inference with a fresh scheduler copy so training state is untouched."""
    original = pipe.scheduler
    pipe.scheduler = type(original).from_config(original.config)
    try:
        yield
    finally:
        pipe.scheduler = original


def cast_trainable_params_to_float32(module: nn.Module) -> None:
    for param in module.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()
