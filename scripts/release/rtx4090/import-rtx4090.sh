#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ "$(uname -m)" == x86_64 ]] || { echo 'ERROR: 此镜像需要 x86_64 主机'; exit 1; }
NAME=SAR_SwinT_RTX4090_CUDA121_Image.tar.gz
sha256sum -c SHA256SUMS
cat "$NAME".part* > "$NAME.tmp"
mv "$NAME.tmp" "$NAME"
sha256sum -c "$NAME.sha256"
docker load -i "$NAME"
[[ "$(docker image inspect sar-swin-p1:rtx4090-cu121 --format '{{.Architecture}}')" == amd64 ]]
echo '镜像已导入。请按 IMPORT.md 检查 4 张 RTX4090 CUDA 环境。'
