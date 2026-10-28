#!/usr/bin/env bash
set -euo pipefail
cd "${1:-sar-amd64-delivery}"
if [[ "$(uname -m)" != x86_64 ]]; then
  echo 'This image requires a Linux x86_64 (amd64) host.' >&2
  exit 1
fi
archive=sar-vlm-qwen3vl4b-image.tar.gz
printf '%s\n' '94562a644f18d522ad0ad4919af7dc5843bf0dd5970a8ceff0e5401d0be43964  sar-vlm-qwen3vl4b-image.tar.gz.part01' | sha256sum --check -
printf '%s\n' 'a334bba83af74eb05dec2628df8247dcfc889dc074551f6f9d7c8f319c98b212  sar-vlm-qwen3vl4b-image.tar.gz.part02' | sha256sum --check -
printf '%s\n' '262cf834f9afed58c922e9d1017833655d8afef4d18be957aaab2e65d537e938  sar-vlm-qwen3vl4b-image.tar.gz.part03' | sha256sum --check -
printf '%s\n' '10f0ffbd0f6988dc14af9f097a4a449bd07807d5d9f4c2e03419513009b1a6ec  sar-vlm-qwen3vl4b-image.tar.gz.part04' | sha256sum --check -
cat "$archive.part01" "$archive.part02" "$archive.part03" "$archive.part04" > "$archive"
printf '%s
' '19fb367f9d38d7625ef7a037daaf47f5bf098b634130552a88bf09447d7cbb2c  sar-vlm-qwen3vl4b-image.tar.gz' | sha256sum --check -
gzip -dc "$archive" | docker load
docker image inspect sar-vlm:qwen3vl4b --format '{{.Os}}/{{.Architecture}}'
