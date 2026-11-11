#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
cat sar-mmrotate-swin-t-rfrcnn-v1.tar.gz.part* > sar-mmrotate-swin-t-rfrcnn-v1.tar.gz
sha256sum -c sar-mmrotate-swin-t-rfrcnn-v1.tar.gz.sha256
docker load -i sar-mmrotate-swin-t-rfrcnn-v1.tar.gz
test "$(docker image inspect sar-mmrotate:swin-t-rfrcnn-v1 --format '{{.Architecture}}')" = amd64
