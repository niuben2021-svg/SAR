#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${1:-sar-arm64-delivery}"
cd "${1:-sar-arm64-delivery}"
base=https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-arm64-cu126-20261028
curl --fail --location --retry 3 --output SHA256SUMS "$base/SHA256SUMS"
while read -r digest filename; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ && "$filename" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 2
  curl --fail --location --retry 3 --continue-at - --output "$filename" "$base/$filename"
done < SHA256SUMS
sha256sum --check SHA256SUMS
