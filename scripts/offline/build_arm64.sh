#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'Build on a native ARM64 Linux host (or use the provided GitHub ARM runner).' >&2
  exit 1
fi
context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT
cp "$root/deploy/Dockerfile.arm64" "$context/Dockerfile"
cp "$root/deploy/requirements-container.txt" "$context/requirements-container.txt"
base="${SAR_PYTHON_BASE:-python:3.11-slim-bookworm}"
docker pull --platform linux/arm64 "$base"
resolved="$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')"
docker build --platform linux/arm64 --build-arg "PYTHON_BASE=$resolved" \
  -t "${SAR_IMAGE_TAG:-sar-vlm:qwen3vl4b-arm64-cu126}" "$context"
