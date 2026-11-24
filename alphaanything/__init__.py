"""AlphaAnything: RGB-to-RGBA adaptation."""
from __future__ import annotations

__all__ = ["AlphaAnythingPipeline"]


def __getattr__(name: str):
    if name == "AlphaAnythingPipeline":
        from alphaanything.pipeline import AlphaAnythingPipeline

        return AlphaAnythingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
