#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?Usage: $0 /ABS/PATH/dataset}"
DATASET="$(realpath "$DATASET")"
IMAGE='sar-swin-p1:arm64-cu126'
docker run --rm --gpus all --network none --shm-size=8g \
 -v "$DATASET:/data/dataset:ro" \
 "$IMAGE" python smoke_test.py --data-root /data/dataset --device cuda --tile 128 --backward
