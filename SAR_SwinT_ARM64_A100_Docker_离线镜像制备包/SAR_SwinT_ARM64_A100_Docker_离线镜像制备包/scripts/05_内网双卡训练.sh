#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:?Usage: $0 /ABS/PATH/dataset /ABS/PATH/output [batch_per_gpu]}"
OUTPUT="${2:?Usage: $0 /ABS/PATH/dataset /ABS/PATH/output [batch_per_gpu]}"
BS="${3:-1}"
DATASET="$(realpath "$DATASET")"; OUTPUT="$(realpath "$OUTPUT")"
[ -f "$OUTPUT/split.json" ] || { echo 'Missing split.json; run 04 first.'; exit 1; }
IMAGE='sar-swin-p1:arm64-cu126'
# Each A100 is 40 GB. Start with batch per GPU=1; try 2 if memory allows.
docker run --rm --gpus all --network none --shm-size=8g \
 -e OMP_NUM_THREADS=4 -e PYTHONUNBUFFERED=1 \
 -v "$DATASET:/data/dataset:ro" -v "$OUTPUT:/workspace/output" \
 "$IMAGE" torchrun --standalone --nproc_per_node=2 train.py \
   --data-root /data/dataset --split /workspace/output/split.json \
   --swin-weights /workspace/weights/swin_t-704ceda3.pth \
   --tile 512 --epochs 24 --batch-size "$BS" --workers 4 \
   --out-dir /workspace/output/runs/swin_p1
