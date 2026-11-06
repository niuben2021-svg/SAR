#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="sar-swin-p1:arm64-cu126"
echo 'HOST:'; uname -m; nvidia-smi
printf '
Docker GPU integration:
'
docker info --format 'Runtimes: {{json .Runtimes}}' || true
docker run --rm --gpus all --network none --shm-size=8g "$IMAGE" \
  python /workspace/sar/check_env.py --require-cuda
echo 'GPU environment OK. Next: scripts/04_内网准备数据.sh DATASET_DIR OUTPUT_DIR'
