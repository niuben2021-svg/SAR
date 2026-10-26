#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
base="${SAR_BASE_IMAGE:-pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime}"
image="${SAR_IMAGE_TAG:-sar-vlm:qwen3vl4b}"
if [[ "${1:-}" == "--help" ]]; then
  echo 'Set SAR_BASE_IMAGE (official PyTorch GPU tag/digest), SAR_IMAGE_TAG; run this script online.'
  exit 0
fi
docker pull --platform linux/amd64 "$base"
# Resolve to immutable digest instead of building from a mutable tag.
resolved="$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')"
if [[ -z "$resolved" || "$resolved" != *@sha256:* ]]; then
  echo 'Cannot resolve a registry digest for the base image' >&2; exit 1
fi
docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$resolved" \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  -t "$image" -f deploy/Dockerfile .
docker run --rm --network none "$image" cpu
# Image inspection records image ID, base digest label, architecture; no credentials.
docker image inspect "$image" --format 'ID={{.Id}} platform={{.Os}}/{{.Architecture}} base={{index .Config.Labels "org.opencontainers.image.base.name"}}'
