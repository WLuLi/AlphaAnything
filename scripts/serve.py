"""Lightweight HTTP inference server keeping one base model resident on GPU.

Optional dependencies (not in requirements.txt):
    pip install fastapi uvicorn python-multipart
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys
import threading

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_checkpoint_spec(spec: str) -> tuple[str | None, str]:
    if "=" in spec and not Path(spec).exists():
        name, path = spec.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --checkpoint specification: {spec!r}. Use PATH or NAME=PATH.")
        return name, path
    return None, spec


def build_app(pipeline):
    import importlib.util

    try:
        from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
    except ImportError as exc:
        raise RuntimeError(
            "Missing server dependencies. Install with: pip install fastapi uvicorn python-multipart"
        ) from exc
    if importlib.util.find_spec("python_multipart") is None and importlib.util.find_spec("multipart") is None:
        raise RuntimeError(
            "Missing server dependencies. Install with: pip install fastapi uvicorn python-multipart"
        )

    app = FastAPI(title="AlphaAnything Server")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "backbone": pipeline.backbone,
            "base_model": pipeline.base_model,
            "adapters": pipeline.adapters,
            "active_adapter": pipeline.active_adapter,
            "alpha_enabled": pipeline.alpha_enabled,
        }

    @app.post("/generate")
    def generate(
        prompt: str | None = Form(None),
        description: str | None = Form(None),
        adapter: str | None = Form(None),
        enable_alpha: bool = Form(True),
        height: int | None = Form(None),
        width: int | None = Form(None),
        num_inference_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        negative_prompt: str | None = Form(None),
        seed: int | None = Form(None),
        image: UploadFile | None = File(None),
    ):
        pil_image = None
        if image is not None:
            raw = image.file.read()
            try:
                pil_image = Image.open(BytesIO(raw)).convert("RGB")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc

        try:
            with lock:
                output = pipeline(
                    prompt=prompt,
                    description=description,
                    image=pil_image,
                    enable_alpha=enable_alpha,
                    adapter=adapter,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    seed=seed,
                )
                resolved_adapter = adapter or pipeline.active_adapter
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        buf = BytesIO()
        output.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "X-Alpha-Enabled": str(bool(enable_alpha)).lower(),
                "X-Adapter": str(resolved_adapter),
                "X-Image-Mode": output.mode,
            },
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve AlphaAnything with runtime RGB/RGBA and adapter switching.")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="One or more checkpoint directories (PATH or NAME=PATH) sharing the same backbone.",
    )
    parser.add_argument("--base-model", help="Optional base model ID or local path overriding adapter_config.json.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Missing server dependencies. Install with: pip install fastapi uvicorn python-multipart"
        ) from exc

    from alphaanything import AlphaAnythingPipeline

    specs = [_parse_checkpoint_spec(item) for item in args.checkpoint]
    first_name, first_path = specs[0]
    pipeline = AlphaAnythingPipeline.from_pretrained(
        first_path,
        base_model=args.base_model,
        adapter_name=first_name,
        device=args.device,
        mixed_precision=args.mixed_precision,
        local_files_only=args.local_files_only,
    )
    for name, path in specs[1:]:
        pipeline.load_adapter(path, name=name)

    app = build_app(pipeline)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
