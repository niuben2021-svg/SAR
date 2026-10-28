#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SAR_IMAGE_TAG="${SAR_IMAGE_TAG:-sar-vlm:qwen3vl4b-arm64-cu126}"
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'This launcher requires a native ARM64 Linux host.' >&2
  exit 1
fi
actual="$(docker image inspect "$SAR_IMAGE_TAG" --format '{{.Os}}/{{.Architecture}}')"
if [[ "$actual" != linux/arm64 ]]; then
  echo "Expected linux/arm64 image, got $actual" >&2
  exit 1
fi
exec bash "$root/scripts/offline/run.sh" "$@"
