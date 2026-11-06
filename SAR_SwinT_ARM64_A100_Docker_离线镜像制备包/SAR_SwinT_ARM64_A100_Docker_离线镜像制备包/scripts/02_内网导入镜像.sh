#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
FILE="${1:-SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz}"
if [ ! -f "$FILE" ]; then
 echo "Image archive not found: $FILE" >&2; exit 1
fi
[ ! -f "${FILE}.sha256" ] || sha256sum -c "${FILE}.sha256"
docker load -i "$FILE"
arch="$(docker image inspect --format '{{.Architecture}}' 'sar-swin-p1:arm64-cu126')"
echo "Image arch: $arch"
[ "$arch" = 'arm64' ] || { echo 'ERROR: non-ARM64 image'; exit 1; }
echo 'Image imported. Next: scripts/03_内网GPU环境检查.sh'
