#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
cat cvmt_cu118.tar.gz.part* > cvmt_cu118.tar.gz
sha256sum -c cvmt_cu118.tar.gz.sha256
docker load -i cvmt_cu118.tar.gz
