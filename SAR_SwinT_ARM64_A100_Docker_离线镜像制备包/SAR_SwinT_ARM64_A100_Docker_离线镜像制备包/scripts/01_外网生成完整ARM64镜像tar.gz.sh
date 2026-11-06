#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="sar-swin-p1:arm64-cu126"
BASE="nvcr.io/nvidia/pytorch:24.10-py3"
ARCHIVE="SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz"
command -v docker >/dev/null || { echo 'ERROR: External machine must have Docker'; exit 1; }
command -v curl >/dev/null || { echo 'ERROR: curl missing'; exit 1; }
# Bundle the official weight into the final image so the intranet only needs this tar.gz.
"scripts/00_外网下载权重.sh"
echo "Pulling ARM64 NGC base image (large download, allow adequate free disk) ..."
docker pull --platform linux/arm64 "$BASE"
echo "Building project image (Dockerfile intentionally has no RUN, so x86_64 hosts can assemble arm64 image) ..."
DOCKER_BUILDKIT=1 docker build --platform linux/arm64 --pull=false -f Dockerfile -t "$IMAGE" .
arch="$(docker image inspect --format '{{.Architecture}}' "$IMAGE")"
if [ "$arch" != 'arm64' ]; then
  echo "ERROR: built image architecture is [$arch], expected arm64" >&2
  exit 1
fi
echo "Saving a full offline Docker image ..."
docker save "$IMAGE" | gzip -1 > "${ARCHIVE}.tmp"
mv "${ARCHIVE}.tmp" "$ARCHIVE"
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
echo 'Done:'
ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"
echo 'Copy these two files to the Kylin ARM64 intranet server.'
