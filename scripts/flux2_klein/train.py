"""Train FLUX.2 Klein text-to-RGBA or image-to-RGBA adapters."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from alphaanything.training.flux2_klein import main

if __name__ == "__main__":
    main()
