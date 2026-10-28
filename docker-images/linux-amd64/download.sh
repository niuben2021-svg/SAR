#!/usr/bin/env bash
set -euo pipefail
out="${1:-sar-amd64-delivery}"
mkdir -p "$out"
cd "$out"
base="https://github.com/niuben2021-svg/SAR/releases/download/qwen3vl4b-offline-20261023"
curl --fail --location --retry 3 --continue-at - --output 'DOWNLOAD_README.md' "$base/DOWNLOAD_README.md"
curl --fail --location --retry 3 --continue-at - --output 'qwen3vl4b-offline-guide.md' "$base/qwen3vl4b-offline-guide.md"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-image.tar.gz.part01' "$base/sar-vlm-qwen3vl4b-image.tar.gz.part01"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-image.tar.gz.part02' "$base/sar-vlm-qwen3vl4b-image.tar.gz.part02"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-image.tar.gz.part03' "$base/sar-vlm-qwen3vl4b-image.tar.gz.part03"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-image.tar.gz.part04' "$base/sar-vlm-qwen3vl4b-image.tar.gz.part04"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-image.tar.gz.sha256' "$base/sar-vlm-qwen3vl4b-image.tar.gz.sha256"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-source.tar.gz' "$base/sar-vlm-qwen3vl4b-source.tar.gz"
curl --fail --location --retry 3 --continue-at - --output 'sar-vlm-qwen3vl4b-source.tar.gz.sha256' "$base/sar-vlm-qwen3vl4b-source.tar.gz.sha256"
curl --fail --location --retry 3 --continue-at - --output 'SHA256SUMS' "$base/SHA256SUMS"
sha256sum --check SHA256SUMS
