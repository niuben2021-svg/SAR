#!/usr/bin/env bash
set -euo pipefail
TAG="${IMAGE_TAG:-sar-swin-p1:rtx4090-cu121}"
echo 'Checking NVIDIA GPU runtime (must have NVIDIA Container Toolkit configured)...'
docker run --rm --gpus all --ipc=host \
  "$TAG" python check_env.py --require-gpus 4 --run-model
