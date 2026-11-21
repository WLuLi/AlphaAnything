from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object


def count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def shard_eval_items(items: Sequence[Any], accelerator: Accelerator) -> Iterator[tuple[int, Any]]:
    for number in range(accelerator.process_index, len(items), accelerator.num_processes):
        yield number, items[number]


def gather_eval_records(
    records: list[dict[str, Any]], accelerator: Accelerator, *, index_key: str = "sample_number",
) -> list[dict[str, Any]]:
    if accelerator.num_processes > 1:
        with torch.inference_mode(False):
            records = gather_object(records)
    return sorted(records, key=lambda record: record[index_key])


def gather_loss_logs(prefix: str, outputs: dict[str, torch.Tensor], accelerator: Accelerator) -> dict[str, float]:
    return {
        f"{prefix}/{key}": accelerator.gather_for_metrics(val.detach().float().reshape(1)).mean().item()
        for key, val in outputs.items()
    }


def create_tensorboard_writer(report_to: str | None, logging_dir: Path, accelerator: Accelerator):
    if report_to not in ("tensorboard", "tb") or not accelerator.is_main_process:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(logging_dir))


def write_tensorboard_scalars(writer: Any, logs: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    for key, value in logs.items():
        if key == "step":
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        if isinstance(value, (int, float, bool)):
            writer.add_scalar(key, float(value), step)
    writer.flush()
