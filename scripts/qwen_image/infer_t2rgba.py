"""Run qwen_image t2rgba inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alphaanything.inference import main

if __name__ == "__main__":
    main("qwen_image", "text_to_rgba")
