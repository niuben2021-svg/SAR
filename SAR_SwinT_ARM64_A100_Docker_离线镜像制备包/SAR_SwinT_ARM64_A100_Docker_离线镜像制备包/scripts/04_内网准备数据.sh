#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?Usage: $0 /ABS/PATH/dataset /ABS/PATH/output}"
OUTPUT="${2:?Usage: $0 /ABS/PATH/dataset /ABS/PATH/output}"
DATASET="$(realpath "$DATASET")"
mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"
IMAGE="sar-swin-p1:arm64-cu126"
docker run --rm --gpus all --network none --shm-size=8g \
 -v "$DATASET:/data/dataset:ro" -v "$OUTPUT:/workspace/output" \
 "$IMAGE" bash -lc 'set -e; \
 python analyze_labels.py --data-root /data/dataset; \
 python prepare.py --data-root /data/dataset --val-ratio 0.15 --seed 42 --out /workspace/output/split.json'
echo "Saved $OUTPUT/split.json"
