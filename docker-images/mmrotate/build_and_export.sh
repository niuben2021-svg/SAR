#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
IMAGE='sar-mmrotate:swin-t-rfrcnn-v1'
OUT="${OUT:-$PWD/sar-mmrotate-swin-t-rfrcnn-v1.tar}"
docker build --platform linux/amd64 --progress=plain -t "$IMAGE" . 2>&1 | tee build.log
docker run --rm --gpus all --ipc=host "$IMAGE" bash /workspace/scripts/smoke_test.sh 2>&1 | tee smoke_test.log
docker save -o "$OUT" "$IMAGE"
sha256sum "$OUT" | tee "$OUT.sha256"
echo "EXPORTED: $OUT"
