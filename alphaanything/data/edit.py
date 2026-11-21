from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from alphaanything.data.rgba import resize_straight_rgba_premultiplied

DEFAULT_EDIT_BUCKET_ASPECT_RATIOS = ("1:1", "16:9", "9:16", "3:2", "2:3")


@dataclass(frozen=True)
class EditBucket:
    key: str
    nominal_ratio: float
    width: int
    height: int

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "nominal_ratio": self.nominal_ratio, "width": self.width, "height": self.height}


def _align(value: float, multiple: int) -> int:
    m = max(1, int(multiple))
    return max(m, int(round(value / m)) * m)


def build_edit_buckets(
    resolution: int, aspect_ratios: Iterable[Any] | None = None, size_multiple: int = 32,
) -> list[EditBucket]:
    res = _align(float(resolution), size_multiple)
    buckets: list[EditBucket] = []
    for value in (aspect_ratios or DEFAULT_EDIT_BUCKET_ASPECT_RATIOS):
        if isinstance(value, str) and ":" in value:
            num, den = (float(p) for p in value.strip().split(":", 1))
            key, ratio = f"{num:g}:{den:g}", num / den
        else:
            ratio = float(value)
            key = f"{ratio:g}"
        w, h = (res, _align(res / ratio, size_multiple)) if ratio >= 1.0 else (_align(res * ratio, size_multiple), res)
        buckets.append(EditBucket(key=key, nominal_ratio=ratio, width=w, height=h))
    return buckets


def _center_crop_box(width: int, height: int, target_ratio: float) -> tuple[int, int, int, int]:
    src_ratio = float(width) / float(height)
    if math.isclose(src_ratio, target_ratio, rel_tol=1e-6, abs_tol=1e-6):
        return 0, 0, height, width
    if src_ratio > target_ratio:
        crop_w = max(1, min(width, int(round(height * target_ratio))))
        return 0, (width - crop_w) // 2, height, crop_w
    crop_h = max(1, min(height, int(round(width / target_ratio))))
    return (height - crop_h) // 2, 0, crop_h, width


def _read_pil_tensor(path: Path, mode: str) -> torch.Tensor:
    with Image.open(path) as image:
        return TF.pil_to_tensor(exif_transpose(image).convert(mode))


class RGBAEditDataset(Dataset):
    """Paired RGB source and RGBA target layer dataset loaded from a JSONL manifest."""

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        self.manifest_path = Path(data_cfg["manifest_path"])
        self.condition_column = str(data_cfg.get("condition_image_column", "condition_image"))
        self.target_column = str(data_cfg.get("target_rgba_column", "target_rgba"))
        self.caption_column = str(data_cfg.get("caption_column", "prompt"))
        self.id_column = str(data_cfg.get("id_column", "id"))
        self.require_same_size = bool(data_cfg.get("require_same_size", True))
        self.resolution = int(data_cfg.get("resolution", 1024))
        self.bucket_size_multiple = int(data_cfg.get("bucket_size_multiple", 32))
        self.buckets = build_edit_buckets(
            self.resolution,
            data_cfg.get("bucket_aspect_ratios", DEFAULT_EDIT_BUCKET_ASPECT_RATIOS),
            self.bucket_size_multiple,
        )
        self.items = self._load_manifest()
        repeats = int(data_cfg.get("repeats", 1))
        if repeats > 1:
            self.items = [dict(item) for item in self.items for _ in range(repeats)]
        if not self.items:
            raise ValueError(f"No edit pairs found in {self.manifest_path}.")

    def _load_manifest(self) -> list[dict[str, Any]]:
        base = self.manifest_path.parent
        items: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                cond = Path(row[self.condition_column])
                tgt = Path(row[self.target_column])
                cond = cond if cond.is_absolute() else base / cond
                tgt = tgt if tgt.is_absolute() else base / tgt
                w, h = int(row.get("canvas_width") or 0), int(row.get("canvas_height") or 0)
                if w <= 0 or h <= 0:
                    with Image.open(cond) as img:
                        w, h = img.size
                        if img.getexif().get(274) in (5, 6, 7, 8):
                            w, h = h, w
                ratio = float(w) / float(h)
                bucket = min(self.buckets, key=lambda b: abs(math.log(ratio / b.nominal_ratio)))
                items.append({
                    "id": str(row[self.id_column]),
                    "condition_path": cond,
                    "target_path": tgt,
                    "prompt": str(row[self.caption_column]),
                    "source_size": (w, h),
                    "bucket_key": bucket.key,
                    "bucket_size": (bucket.width, bucket.height),
                })
        return items

    def bucket_counts(self) -> dict[str, int]:
        counts = {b.key: 0 for b in self.buckets}
        for item in self.items:
            counts[item["bucket_key"]] += 1
        return counts

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        cond = _read_pil_tensor(item["condition_path"], "RGB")
        tgt = _read_pil_tensor(item["target_path"], "RGBA")
        if self.require_same_size and cond.shape[-2:] != tgt.shape[-2:]:
            raise ValueError(f"Paired image sizes differ for {item['id']}: {tuple(cond.shape[-2:])} vs {tuple(tgt.shape[-2:])}")

        bw, bh = item["bucket_size"]
        out_ratio = float(bw) / float(bh)
        top, left, ch, cw = _center_crop_box(cond.shape[-1], cond.shape[-2], out_ratio)
        t_top, t_left, t_ch, t_cw = _center_crop_box(tgt.shape[-1], tgt.shape[-2], out_ratio)
        cond = TF.crop(cond, top=top, left=left, height=ch, width=cw)
        tgt = TF.crop(tgt, top=t_top, left=t_left, height=t_ch, width=t_cw)

        cond = TF.resize(cond, [bh, bw], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        cond = TF.convert_image_dtype(cond, torch.float32) * 2.0 - 1.0
        tgt = resize_straight_rgba_premultiplied(TF.convert_image_dtype(tgt, torch.float32), [bh, bw]) * 2.0 - 1.0

        return {
            "condition_pixel_values": cond,
            "target_pixel_values": tgt,
            "prompt": item["prompt"],
            "id": item["id"],
            "condition_path": str(item["condition_path"]),
            "target_path": str(item["target_path"]),
            "bucket_key": item["bucket_key"],
            "bucket_size": item["bucket_size"],
        }


class EditBucketBatchSampler:
    def __init__(
        self, dataset: RGBAEditDataset, batch_size: int, shuffle: bool, drop_last: bool = False, seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.items):
            grouped[item["bucket_key"]].append(index)
        self.buckets = dict(grouped)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for indices in self.buckets.values():
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)
            for start in range(0, len(idxs), self.batch_size):
                batch = idxs[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(
            len(idxs) // self.batch_size if self.drop_last else math.ceil(len(idxs) / self.batch_size)
            for idxs in self.buckets.values()
        )


def collate_edit_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "condition_pixel_values": torch.stack([e["condition_pixel_values"] for e in examples]).float(),
        "target_pixel_values": torch.stack([e["target_pixel_values"] for e in examples]).float(),
        "prompts": [e["prompt"] for e in examples],
        "ids": [e["id"] for e in examples],
        "condition_paths": [e["condition_path"] for e in examples],
        "target_paths": [e["target_path"] for e in examples],
        "bucket_keys": [e["bucket_key"] for e in examples],
        "bucket_sizes": [e["bucket_size"] for e in examples],
    }


def build_edit_loader(
    dataset: RGBAEditDataset, train_cfg: dict[str, Any], data_cfg: dict[str, Any], *, shuffle: bool,
) -> DataLoader:
    num_workers = int(data_cfg.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_sampler=EditBucketBatchSampler(
            dataset,
            batch_size=int(train_cfg.get("train_batch_size", 1)),
            shuffle=shuffle,
            drop_last=bool(train_cfg.get("drop_last", False)),
            seed=int(train_cfg.get("seed", 0)),
        ),
        num_workers=num_workers,
        collate_fn=collate_edit_examples,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)) and num_workers > 0,
    )
