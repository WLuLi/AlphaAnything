#!/usr/bin/env python3
"""Reconstruct PrismLayersPro source RGB and full-canvas target RGBA images from Parquet files."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import gzip
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def _open_annotations(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _decode_parquet_image(value: Any, prism_dir: Path) -> Image.Image:
    if not isinstance(value, dict):
        raise ValueError("Expected a Parquet image struct with 'bytes' or 'path'.")
    if value.get("bytes"):
        source: Any = io.BytesIO(value["bytes"])
    elif value.get("path"):
        source = prism_dir / value["path"]
    else:
        raise ValueError("Missing image bytes/path in Parquet cell.")
    with Image.open(source) as img:
        img.load()
        return img.copy()


def _place_layer_on_canvas(
    layer_img: Image.Image,
    box_xyxy: list[int],
    canvas_size: tuple[int, int],
) -> Image.Image:
    x0, y0, x1, y1 = map(int, box_xyxy)
    width, height = x1 - x0, y1 - y0
    resized = layer_img.convert("RGBA").resize((width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    # Paste without a mask so straight RGBA alpha is preserved rather than squared.
    canvas.paste(resized, (x0, y0))
    return canvas


def _process_parquet(
    parquet_rel: str,
    row_specs: dict[int, dict[str, Any]],
    prism_dir: Path,
    output_dir: Path,
    *,
    download: bool,
    skip_existing: bool,
) -> tuple[int, int]:
    parquet_path = prism_dir / parquet_rel
    if not parquet_path.is_file():
        if not download:
            raise FileNotFoundError(
                f"Missing {parquet_path}. Download artplus/PrismLayersPro or pass --download."
            )
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="artplus/PrismLayersPro",
            repo_type="dataset",
            filename=parquet_rel,
            local_dir=str(prism_dir),
        )

    needed_layers = sorted({layer_idx for spec in row_specs.values() for layer_idx in spec["layers"]})
    columns = ["whole_image"] + [f"layer_{idx:02d}" for idx in needed_layers]

    pf = pq.ParquetFile(parquet_path)
    saved_sources = 0
    saved_layers = 0
    row_offset = 0

    for rg_idx in range(pf.num_row_groups):
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_end = row_offset + rg_rows
        local_indices = [r for r in row_specs if row_offset <= r < rg_end]
        if not local_indices:
            row_offset = rg_end
            continue

        table = pf.read_row_group(rg_idx, columns=columns)
        for global_row in local_indices:
            spec = row_specs[global_row]
            local_row = global_row - row_offset
            canvas_size = (int(spec["canvas_width"]), int(spec["canvas_height"]))

            src_dst = output_dir / spec["condition_image"]
            if not (skip_existing and src_dst.is_file()):
                src_img = _decode_parquet_image(table["whole_image"][local_row].as_py(), prism_dir).convert("RGB")
                src_dst.parent.mkdir(parents=True, exist_ok=True)
                src_img.save(src_dst)
                saved_sources += 1

            for layer_idx, (target_rel, box_xyxy) in spec["layers"].items():
                tgt_dst = output_dir / target_rel
                if skip_existing and tgt_dst.is_file():
                    continue
                col_name = f"layer_{layer_idx:02d}"
                layer_img = _decode_parquet_image(table[col_name][local_row].as_py(), prism_dir)
                canvas = _place_layer_on_canvas(layer_img, box_xyxy, canvas_size)
                tgt_dst.parent.mkdir(parents=True, exist_ok=True)
                canvas.save(tgt_dst)
                saved_layers += 1

        row_offset = rg_end

    return saved_sources, saved_layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/train/prismlayerspro_annotations.jsonl"),
        help="Path to prismlayerspro_annotations.jsonl (or .jsonl.gz).",
    )
    parser.add_argument(
        "--prism-dir",
        type=Path,
        required=True,
        help="Local directory containing artplus/PrismLayersPro (with data/*.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/train/prismlayerspro"),
        help="Directory to write extracted images and train.jsonl.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing Parquet files from artplus/PrismLayersPro on Hugging Face.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of Parquet files to process in parallel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already extracted PNG files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of annotation records to process.",
    )
    args = parser.parse_args()

    by_parquet: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    manifest_rows: list[dict[str, Any]] = []

    with _open_annotations(args.annotations) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parquet_rel = row["source_ref"]["parquet"]
            row_index = int(row["source_ref"]["row_index"])
            spec = by_parquet[parquet_rel].setdefault(
                row_index,
                {
                    "condition_image": row["condition_image"],
                    "canvas_width": int(row["canvas_width"]),
                    "canvas_height": int(row["canvas_height"]),
                    "layers": {},
                },
            )
            spec["layers"][int(row["layer_index"])] = (row["target_rgba"], row["box_xyxy"])
            manifest_rows.append(
                {
                    "id": row["id"],
                    "condition_image": row["condition_image"],
                    "target_rgba": row["target_rgba"],
                    "prompt": row["prompt"],
                    "canvas_width": int(row["canvas_width"]),
                    "canvas_height": int(row["canvas_height"]),
                }
            )
            if args.limit is not None and len(manifest_rows) >= args.limit:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parquet_items = sorted(by_parquet.items())
    total_sources = 0
    total_layers = 0

    if args.num_workers <= 1:
        for parquet_rel, row_specs in tqdm(parquet_items, desc="Extracting PrismLayersPro"):
            s, l = _process_parquet(
                parquet_rel,
                row_specs,
                args.prism_dir,
                args.output_dir,
                download=args.download,
                skip_existing=not args.overwrite,
            )
            total_sources += s
            total_layers += l
    else:
        with futures.ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futs = [
                pool.submit(
                    _process_parquet,
                    parquet_rel,
                    row_specs,
                    args.prism_dir,
                    args.output_dir,
                    download=args.download,
                    skip_existing=not args.overwrite,
                )
                for parquet_rel, row_specs in parquet_items
            ]
            for fut in tqdm(futures.as_completed(futs), total=len(futs), desc="Extracting PrismLayersPro"):
                s, l = fut.result()
                total_sources += s
                total_layers += l

    manifest_path = args.output_dir / "train.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Done: wrote {total_sources} source images, {total_layers} target RGBA layers, "
        f"and {len(manifest_rows)} records to {manifest_path}"
    )


if __name__ == "__main__":
    main()
