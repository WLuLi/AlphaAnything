from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import functional as TF

from alphaanything.data.rgba import resize_straight_rgba_premultiplied

DEFAULT_BUCKET_ASPECT_RATIOS = ("1:1", "16:9", "9:16", "3:2", "2:3")


def _parse_aspect_ratio(value: Any) -> tuple[str, float]:
    if isinstance(value, str) and ":" in value:
        num, den = (float(p) for p in value.strip().split(":", 1))
        return f"{num:g}:{den:g}", num / den
    ratio = float(value)
    return f"{ratio:g}", ratio


class RGBACaptionDataset(Dataset):
    """Text-to-RGBA dataset loaded from a UTF-8 caption CSV."""

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        self.dataset_name = data_cfg["dataset_name"]
        self.image_column = data_cfg.get("image_column", "image")
        self.caption_column = data_cfg.get("caption_column", "caption")
        self.short_caption_column = data_cfg.get("short_caption_column")
        self.short_caption_probability = float(data_cfg.get("short_caption_probability", 0.0))
        self.implicit_caption_column = data_cfg.get("implicit_caption_column")
        self.implicit_short_caption_column = data_cfg.get("implicit_short_caption_column")
        self.implicit_caption_probability = float(data_cfg.get("implicit_caption_probability", 0.0))
        self.size = int(data_cfg.get("resolution", 1024))
        self.bucket_by_aspect_ratio = bool(data_cfg.get("bucket_by_aspect_ratio", True))
        self.bucket_size_multiple = int(data_cfg.get("bucket_size_multiple", 32))
        bucket_ratios = data_cfg.get("bucket_aspect_ratios", "auto")
        self.auto_bucket = self.bucket_by_aspect_ratio and isinstance(bucket_ratios, str) and bucket_ratios.lower() == "auto"
        self.bucket_specs = (
            [_parse_aspect_ratio(v) for v in (bucket_ratios or DEFAULT_BUCKET_ASPECT_RATIOS)]
            if self.bucket_by_aspect_ratio and not self.auto_bucket
            else []
        )
        self.bucket_aspect_ratio_keys = [key for key, _ in self.bucket_specs]
        self.random_flip = bool(data_cfg.get("random_flip", False))

        self.items = self._load_csv()
        repeats = int(data_cfg.get("repeats", 1))
        if repeats > 1:
            self.items = [dict(item) for item in self.items for _ in range(repeats)]
        if not self.items:
            raise ValueError(f"No training samples found in {self.dataset_name}.")
        if self.bucket_by_aspect_ratio:
            self._assign_bucket_sizes()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])

    def _load_csv(self) -> list[dict[str, Any]]:
        path = Path(self.dataset_name)
        csv_path = path if path.suffix.lower() == ".csv" else path / "train.csv"
        with csv_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        cols = set(rows[0].keys())
        base = csv_path.parent
        items = []
        for row in rows:
            img_path = Path(row[self.image_column])
            if not img_path.is_absolute():
                img_path = base / img_path
            item: dict[str, Any] = {"image": str(img_path), "prompt": row.get(self.caption_column, "")}
            if self.short_caption_column in cols and row.get(self.short_caption_column):
                item["short_prompt"] = row[self.short_caption_column]
            if self.implicit_caption_column in cols and row.get(self.implicit_caption_column):
                item["implicit_prompt"] = row[self.implicit_caption_column]
            if self.implicit_short_caption_column in cols and row.get(self.implicit_short_caption_column):
                item["implicit_short_prompt"] = row[self.implicit_short_caption_column]
            if row.get("width") and row.get("height"):
                item["source_size"] = (int(row["width"]), int(row["height"]))
            items.append(item)
        return items

    def _bucket_size(self, ratio: float) -> tuple[int, int]:
        raw_w, raw_h = (float(self.size), float(self.size) / ratio) if ratio >= 1.0 else (float(self.size) * ratio, float(self.size))
        m = max(1, self.bucket_size_multiple)
        return max(m, int(round(raw_w / m)) * m), max(m, int(round(raw_h / m)) * m)

    def _item_source_size(self, item: dict[str, Any]) -> tuple[int, int]:
        if "source_size" in item:
            return item["source_size"]
        with Image.open(item["image"]) as image:
            w, h = image.size
            if image.getexif().get(274) in (5, 6, 7, 8):
                w, h = h, w
            return int(w), int(h)

    def _assign_bucket_sizes(self) -> None:
        for item in self.items:
            w, h = self._item_source_size(item)
            src_ratio = float(w) / float(h)
            if self.auto_bucket:
                size = self._bucket_size(src_ratio)
                key, ratio = f"{size[0]}x{size[1]}", src_ratio
            else:
                key, ratio = min(self.bucket_specs, key=lambda spec: abs(math.log(src_ratio / spec[1])))
                size = self._bucket_size(ratio)
            item["bucket_aspect_ratio"] = key
            item["bucket_ratio"] = ratio
            item["bucket_size"] = size
        if self.auto_bucket:
            self.bucket_aspect_ratio_keys = sorted({item["bucket_aspect_ratio"] for item in self.items})

    def bucket_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for item in self.items:
            if "bucket_aspect_ratio" in item:
                counts[item["bucket_aspect_ratio"]] += 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def _read_image(self, image_path: str) -> torch.Tensor:
        try:
            return decode_image(image_path, mode=ImageReadMode.RGBA, apply_exif_orientation=True)
        except (RuntimeError, OSError, ValueError):
            with Image.open(image_path) as image:
                return TF.pil_to_tensor(exif_transpose(image).convert("RGBA"))

    @staticmethod
    def _center_crop_to_ratio(image: torch.Tensor, target_ratio: float) -> torch.Tensor:
        _, h, w = image.shape
        src_ratio = float(w) / float(h)
        if math.isclose(src_ratio, target_ratio, rel_tol=1e-6, abs_tol=1e-6):
            return image
        if src_ratio > target_ratio:
            crop_w = max(1, min(w, int(round(h * target_ratio))))
            return TF.crop(image, top=0, left=(w - crop_w) // 2, height=h, width=crop_w)
        crop_h = max(1, min(h, int(round(w / target_ratio))))
        return TF.crop(image, top=(h - crop_h) // 2, left=0, height=crop_h, width=w)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = self._read_image(item["image"])
        bucket_size = item.get("bucket_size")
        if bucket_size is not None:
            w, h = bucket_size
            if not self.auto_bucket:
                image = self._center_crop_to_ratio(image, item["bucket_ratio"])
            rgba = resize_straight_rgba_premultiplied(TF.convert_image_dtype(image, torch.float32), [h, w])
        else:
            rgba = resize_straight_rgba_premultiplied(TF.convert_image_dtype(image, torch.float32), [self.size, self.size])
        if self.random_flip and random.random() < 0.5:
            rgba = TF.hflip(rgba)
        rgba = self.normalize(rgba)

        use_implicit = bool(item.get("implicit_prompt")) and (random.random() < self.implicit_caption_probability)
        prompt = item["implicit_prompt"] if use_implicit else item["prompt"]
        short_prompt = item.get("implicit_short_prompt") if use_implicit else item.get("short_prompt")
        if short_prompt and random.random() < self.short_caption_probability:
            prompt = short_prompt

        return {
            "pixel_values": rgba,
            "prompt": prompt,
            "path": item["image"],
            "bucket_size": bucket_size,
            "bucket_aspect_ratio": item.get("bucket_aspect_ratio"),
        }


class BucketBatchSampler:
    def __init__(
        self, dataset: RGBACaptionDataset, batch_size: int, shuffle: bool, drop_last: bool = False, seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        buckets: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(dataset.items):
            buckets[item["bucket_aspect_ratio"]].append(index)
        self.buckets = dict(buckets)

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


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]).float(),
        "prompts": [e["prompt"] for e in examples],
        "paths": [e["path"] for e in examples],
        "bucket_sizes": [e.get("bucket_size") for e in examples],
        "bucket_aspect_ratios": [e.get("bucket_aspect_ratio") for e in examples],
    }


def build_loader(dataset: RGBACaptionDataset, train_cfg: dict[str, Any], data_cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    batch_size = int(train_cfg.get("train_batch_size", 1))
    drop_last = bool(train_cfg.get("drop_last", False))
    num_workers = int(data_cfg.get("num_workers", 0))
    if dataset.bucket_by_aspect_ratio:
        return DataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=int(train_cfg.get("seed", 0)),
            ),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=drop_last,
    )
