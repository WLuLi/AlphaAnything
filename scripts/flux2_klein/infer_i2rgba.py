"""Run flux2_klein i2rgba inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alphaanything.inference import main

if __name__ == "__main__":
    main("flux2_klein", "image_to_rgba")
