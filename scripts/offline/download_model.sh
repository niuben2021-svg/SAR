#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo 'Usage: download_model.sh DESTINATION [modelscope|huggingface]'; exit 2; fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$1"
destination="$(realpath -e "$1")"
# This is the external-network download command; regular inner-network runs use network none.
docker run --rm --pull never \
  --mount "type=bind,source=$root,target=/opt/sar-vlm,readonly" \
  --mount "type=bind,source=$destination,target=/downloads/qwen3vl4b" \
  --env-file "$root/deploy/offline.env" \
  --workdir /opt/sar-vlm --env PYTHONPATH=/opt/sar-vlm/src \
  --entrypoint /bin/bash "${SAR_IMAGE_TAG:-sar-vlm:qwen3vl4b}" \
  /opt/sar-vlm/deploy/entrypoint.sh download-model \
  --source "${2:-modelscope}" --destination /downloads/qwen3vl4b
