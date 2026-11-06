#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?Usage: $0 /ABS/PATH/dataset /ABS/PATH/output [val|test]}"
OUTPUT="${2:?Usage: $0 /ABS/PATH/dataset /ABS/PATH/output [val|test]}"
SUBSET="${3:-val}"
DATASET="$(realpath "$DATASET")"; OUTPUT="$(realpath "$OUTPUT")"
IMAGE='sar-swin-p1:arm64-cu126'
docker run --rm --gpus all --network none --shm-size=8g \
 -v "$DATASET:/data/dataset:ro" -v "$OUTPUT:/workspace/output" \
 "$IMAGE" python evaluate.py \
   --checkpoint /workspace/output/runs/swin_p1/last.pth \
   --data-root /data/dataset --split /workspace/output/split.json \
   --subset "$SUBSET" --overlap 128 --score-thresh 0.05 \
   --out "/workspace/output/metrics_${SUBSET}.json"
