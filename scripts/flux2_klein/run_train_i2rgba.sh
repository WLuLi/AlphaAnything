#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

accelerate launch \
  --config_file configs/distributed/ddp_8gpu.yaml \
  --num_processes 8 \
  scripts/flux2_klein/train.py \
  --config configs/flux2_klein/train_base_9b_i2rgba.yaml \
  --manifest data/train/prismlayerspro/train.jsonl \
  --output_dir outputs/train/flux2_klein_base_9b_i2rgba \
  --training.per_device_batch_size 1 \
  --training.gradient_accumulation_steps 2 \
  --training.max_train_steps 20000 \
  --seed 42
