#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/source" "$ROOT/weights"
git clone --depth 1 --branch v8.4.209 https://github.com/ultralytics/ultralytics.git "$ROOT/source/ultralytics"
[[ "$(git -C "$ROOT/source/ultralytics" rev-parse HEAD)" == 9b57ed7e6fd366a95047aaa0526f542a62b202f1 ]]
for size in n s m l x; do
  curl -fL --retry 4 --connect-timeout 30 --max-time 600 \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11${size}-obb.pt" \
    -o "$ROOT/weights/yolo11${size}-obb.pt"
done
(cd "$ROOT/weights" && sha256sum *.pt) > "$ROOT/weights/SHA256SUMS"
