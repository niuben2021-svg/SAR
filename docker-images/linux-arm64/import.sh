#!/usr/bin/env bash
set -euo pipefail
cd "${1:-sar-arm64-delivery}"
[[ "$(uname -m)" == aarch64 ]] || { echo 'Requires ARM64 Linux host' >&2; exit 1; }
sha256sum --check SHA256SUMS
archive=sar-vlm-qwen3vl4b-arm64-cu126.tar.gz
shopt -s nullglob
parts=("$archive".part[0-9][0-9])
[[ ${#parts[@]} -gt 0 ]] || { echo 'No archive parts found' >&2; exit 1; }
for ((i=0;i<${#parts[@]};i++)); do
  printf -v expected '%s.part%02d' "$archive" "$((i+1))"
  [[ "${parts[i]}" == "$expected" ]] || { echo 'Missing or misnumbered part' >&2; exit 1; }
done
cat "${parts[@]}" > "$archive"
sha256sum --check "$archive.sha256"
gzip -t "$archive"
gzip -dc "$archive" | docker load
docker image inspect sar-vlm:qwen3vl4b-arm64-cu126 --format '{{.Os}}/{{.Architecture}}'
