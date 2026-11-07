#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT="${1:?Usage: bash scripts/05_train_4x4090.sh /absolute/dataset /absolute/output [per_gpu_batch=1]}"
OUT_ROOT="${2:?Usage: bash scripts/05_train_4x4090.sh /absolute/dataset /absolute/output [per_gpu_batch=1]}"
BATCH="${3:-1}"
[[ -f "$OUT_ROOT/split.json" ]] || { echo "ERROR: Missing $OUT_ROOT/split.json. Run prepare first." >&2; exit 1; }
TAG="${IMAGE_TAG:-sar-swin-p1:rtx4090-cu121}"
mkdir -p "$OUT_ROOT/runs"
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e OMP_NUM_THREADS=4 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v "$DATA_ROOT:/data/dataset:ro" \
  -v "$OUT_ROOT:/workspace/output" \
  "$TAG" \
  torchrun --standalone --nproc_per_node=4 train.py \
  --data-root /data/dataset \
  --split /workspace/output/split.json \
  --swin-weights /workspace/weights/swin_t-704ceda3.pth \
  --tile 512 --epochs 24 --batch-size "$BATCH" --workers 4 \
  --out-dir /workspace/output/runs/swin_p1_4gpu
