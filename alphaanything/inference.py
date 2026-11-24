"""Shared CLI for text-to-RGBA and image-to-RGBA inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaanything.checkpoint import EXTRACTION_PREFIX, read_adapter_config


def resolve_input(args, task: str) -> tuple[str, Path | None]:
    image_conditioned = task == "image_to_rgba"
    prompt = args.prompt
    if not prompt and args.description:
        if not image_conditioned:
            raise ValueError("Descriptions are only supported by image-to-RGBA; use --prompt for text-to-RGBA.")
        prompt = f"{EXTRACTION_PREFIX} {str(args.description).strip()}"
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing prompt.")

    image = None
    if image_conditioned:
        if not args.image:
            raise ValueError("Image-to-RGBA requires --image.")
        image = Path(args.image)
    return prompt, image


def main(backbone: str, task: str) -> None:
    parser = argparse.ArgumentParser(description=f"AlphaAnything {backbone} {task}")
    parser.add_argument("--checkpoint", required=True, help="Exported adapter directory.")
    parser.add_argument("--base-model", help="Optional base model ID or local path overriding adapter_config.json.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt", help="Full prompt passed verbatim to the model.")
    inputs.add_argument("--description", help="Object description; adds the standard extraction prefix.")
    parser.add_argument("--image", help="Condition image for image-to-RGBA.")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--disable-alpha", action="store_true", help="Disable LoRA and alpha branch to run base RGB generation.")
    args = parser.parse_args()

    if task == "image_to_rgba" and (args.height is not None or args.width is not None):
        parser.error("Image-to-RGBA preserves the source aspect ratio near 1024² pixels; omit height/width.")

    config = read_adapter_config(args.checkpoint)
    if config["backbone"] != backbone or config["task"] != task:
        parser.error(f"Checkpoint is {config['backbone']}/{config['task']}, not {backbone}/{task}.")

    params = dict(config["inference"])
    for key in ("height", "width", "num_inference_steps", "guidance_scale", "negative_prompt", "seed", "mixed_precision"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    if args.base_model:
        config["base_model"] = args.base_model

    prompt, image_path = resolve_input(args, task)
    directory = Path(args.output_dir)

    from alphaanything.pipeline import AlphaAnythingPipeline
    from alphaanything.utils.image import save_infer_sample, save_normalized_rgb

    enable_alpha = not args.disable_alpha
    pipeline = AlphaAnythingPipeline.from_pretrained(
        args.checkpoint,
        base_model=args.base_model,
        device=args.device,
        mixed_precision=args.mixed_precision,
        local_files_only=args.local_files_only,
        enable_alpha=enable_alpha,
    )

    directory.mkdir(parents=True, exist_ok=True)
    seed = int(params["seed"])
    rgb, alpha = pipeline.generate_tensors(
        prompt=prompt,
        image=image_path,
        enable_alpha=enable_alpha,
        height=args.height,
        width=args.width,
        num_inference_steps=int(params["num_inference_steps"]),
        guidance_scale=float(params["guidance_scale"]),
        negative_prompt=params["negative_prompt"],
        seed=seed,
    )
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    if alpha is not None:
        paths = {"rgba": directory / "sample.png"}
        paths.update({name: directory / "previews" / name / "sample.png" for name in ("rgb", "alpha", "checker")})
        save_infer_sample(directory, "sample", rgb, alpha, paths=paths)
    else:
        save_normalized_rgb(directory / "sample.png", rgb)
    print(f"Saved {directory / 'sample.png'}")

    metadata = {
        "backbone": backbone,
        "task": task,
        "base_model": config["base_model"],
        "checkpoint_step": config.get("training_step"),
        "alpha_enabled": enable_alpha,
        "inference": params,
        "prompt": prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "output": "sample.png",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved output to {directory}")
