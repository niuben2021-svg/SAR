#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: bash scripts/06_eval.sh /absolute/dataset /absolute/output [val|test]}"
OUT_ROOT="${2:?Usage: bash scripts/06_eval.sh /absolute/dataset /absolute/output [val|test]}"
SUBSET="${3:-val}"
[[ "$SUBSET" == val || "$SUBSET" == test ]] || { echo 'ERROR: subset must be val or test' >&2; exit 1; }
TAG="${IMAGE_TAG:-sar-swin-p1:rtx4090-cu121}"
docker run --rm --gpus all --shm-size=8g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$DATA_ROOT:/data/dataset:ro" \
  -v "$OUT_ROOT:/workspace/output" \
  "$TAG" python evaluate.py \
  --checkpoint /workspace/output/runs/swin_p1_4gpu/last.pth \
  --data-root /data/dataset \
  --split /workspace/output/split.json \
  --subset "$SUBSET" --overlap 128 --score-thresh 0.05 \
  --out "/workspace/output/metrics_${SUBSET}.json"
