#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ "$(uname -m)" == x86_64 ]] || { echo 'Requires x86_64'; exit 1; }
NAME=yolo11_obb_cuda12.1_amd64.tar.gz
sha256sum -c SHA256SUMS
cat "$NAME".part* > "$NAME.tmp"
mv "$NAME.tmp" "$NAME"
sha256sum -c "$NAME.sha256"
docker load -i "$NAME"
[[ "$(docker image inspect sar-yolo11-obb:8.4.209-cu121 --format '{{.Architecture}}')" == amd64 ]]
echo 'Image imported. See IMPORT.md for GPU verification and training.'
