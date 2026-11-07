#!/usr/bin/env bash
set -euo pipefail
IMAGE_FILE="${1:?Usage: bash scripts/02_import_offline.sh /path/to/image.tar.gz}"
if [[ -f "${IMAGE_FILE}.sha256" ]]; then
  (cd "$(dirname "$IMAGE_FILE")" && sha256sum -c "$(basename "$IMAGE_FILE").sha256")
else
  echo 'WARNING: checksum file not found; verify transfer manually.'
fi
docker load -i "$IMAGE_FILE"
docker image inspect sar-swin-p1:rtx4090-cu121 --format 'Image: {{.RepoTags}} architecture: {{.Architecture}}'
