# Data Guide

Download the AlphaAnything training dataset repository from Hugging Face into `data/` at the repository root. All commands below assume the repository root as the working directory.

## 1. Text-to-RGBA Training Data (`data/train/t2rgba_train`)

Extract both TAR shards into `data/train/`:

```bash
tar -xf data/train/t2rgba_train-00000-of-00002.tar -C data/train
tar -xf data/train/t2rgba_train-00001-of-00002.tar -C data/train
```

This produces `data/train/t2rgba_train/train.csv` and `data/train/t2rgba_train/images/000000.png` ... `008962.png`, matching the default path in `configs/flux2_klein/train_base_9b_t2rgba.yaml`.

### Custom Text-to-RGBA CSV format

Use a UTF-8 CSV with `image` and `caption` columns (plus optional `short_caption`, `implicit_caption`, `implicit_short_caption`, `width`, `height`):

```csv
image,caption
images/000000.png,"A delicate glass butterfly with translucent wings."
images/000001.png,"A single translucent crystal flower."
```

Image paths are resolved relative to the CSV's directory. Images should be RGBA PNGs with an intact alpha channel.

## 2. Image-to-RGBA Training Data (`data/train/prismlayerspro`)

We release the text instructions and layer reference metadata in `data/train/prismlayerspro_annotations.jsonl`. Images are reconstructed locally from the upstream [artplus/PrismLayersPro](https://huggingface.co/datasets/artplus/PrismLayersPro) Parquet files:

```bash
# Option A: download missing Parquet files automatically from Hugging Face
python tools/prepare_prismlayerspro.py \
  --annotations data/train/prismlayerspro_annotations.jsonl \
  --prism-dir data/train/PrismLayersPro_raw \
  --output-dir data/train/prismlayerspro \
  --download \
  --num-workers 4

# Option B: use an already downloaded local copy of artplus/PrismLayersPro
python tools/prepare_prismlayerspro.py \
  --annotations data/train/prismlayerspro_annotations.jsonl \
  --prism-dir /path/to/PrismLayersPro \
  --output-dir data/train/prismlayerspro \
  --num-workers 4
```

This extracts all referenced source RGB images, pastes each foreground RGBA layer onto its full 1024×1024 transparent canvas, and writes `data/train/prismlayerspro/train.jsonl` matching `configs/flux2_klein/train_base_9b_i2rgba.yaml`.

### Custom Image-to-RGBA JSONL format

Use a UTF-8 JSONL file with one aligned RGB/RGBA pair per line:

```jsonl
{"id":"glass-001","condition_image":"sources/cocktail.png","target_rgba":"layers/glass.png","prompt":"Extract the cocktail glass, preserving its fine details and transparency."}
```

Image paths are resolved relative to the JSONL's directory. Source and target images should share the same canvas size and spatial alignment.
