#!/usr/bin/env bash
set -euo pipefail
TAR="${1:?Usage: bash import_and_run.sh /path/to/sar-mmrotate-swin-t-rfrcnn-v1.tar /path/to/SAR_DATASET}"
DATA="${2:?Provide dataset root with train/val/test subdirectories}"
docker load -i "$TAR"
mkdir -p "$PWD/mmrotate_outputs"
docker run --gpus all --ipc=host --shm-size=16g -it --rm \
  -v "$DATA:/data/sar:ro" \
  -v "$PWD/mmrotate_outputs:/workspace/work_dirs" \
  -v "$PWD:/workspace/local" \
  -e SAR_CLASSES="${SAR_CLASSES:-airplane}" \
  -e BATCH_PER_GPU="${BATCH_PER_GPU:-2}" \
  -e WORKERS_PER_GPU="${WORKERS_PER_GPU:-12}" \
  -e SWIN_CKPT="${SWIN_CKPT:-}" \
  sar-mmrotate:swin-t-rfrcnn-v1 bash /workspace/scripts/train_4gpu.sh
