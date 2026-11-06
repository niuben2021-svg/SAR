#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NAME=SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz
sha256sum -c SHA256SUMS
cat "$NAME".part* > "$NAME.tmp"
mv "$NAME.tmp" "$NAME"
sha256sum -c "$NAME.sha256"
docker load -i "$NAME"
[[ "$(docker image inspect sar-swin-p1:arm64-cu126 --format '{{.Architecture}}')" == arm64 ]]
echo '镜像已导入。请执行 IMPORT.md 中的双 A100 CUDA 检查。'
