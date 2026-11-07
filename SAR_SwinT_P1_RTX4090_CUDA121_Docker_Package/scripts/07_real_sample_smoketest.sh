#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: bash scripts/07_real_sample_smoketest.sh /absolute/dataset}"
TAG="${IMAGE_TAG:-sar-swin-p1:rtx4090-cu121}"
docker run --rm --gpus all --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$DATA_ROOT:/data/dataset:ro" \
  "$TAG" python smoke_test.py --data-root /data/dataset --device cuda --tile 128 --backward
