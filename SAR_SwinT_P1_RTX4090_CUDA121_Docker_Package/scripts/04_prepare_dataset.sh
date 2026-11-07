#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: bash scripts/04_prepare_dataset.sh /absolute/dataset /absolute/output}"
OUT_ROOT="${2:?Usage: bash scripts/04_prepare_dataset.sh /absolute/dataset /absolute/output}"
mkdir -p "$OUT_ROOT"
TAG="${IMAGE_TAG:-sar-swin-p1:rtx4090-cu121}"
docker run --rm --shm-size=8g \
  -v "$DATA_ROOT:/data/dataset:ro" \
  -v "$OUT_ROOT:/workspace/output" \
  "$TAG" bash -lc 'python analyze_labels.py --data-root /data/dataset && python prepare.py --data-root /data/dataset --val-ratio 0.15 --seed 42 --out /workspace/output/split.json'
