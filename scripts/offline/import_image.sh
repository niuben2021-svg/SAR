#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then echo 'Usage: import_image.sh IMAGE.tar.gz'; exit 2; fi
archive="$1"
(
  cd "$(dirname "$archive")"
  sha256sum -c "$(basename "$archive").sha256"
)
gzip -dc "$archive" | docker load
