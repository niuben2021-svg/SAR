#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base="${SAR_PYTHON_BASE:-python:3.11-slim-bookworm}"
image="${SAR_IMAGE_TAG:-sar-vlm:qwen3vl4b}"
context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT
docker pull --platform linux/amd64 "$base"
resolved="$(docker image inspect "$base" --format '{{index .RepoDigests 0}}')"
cp "$root/deploy/Dockerfile.runtime" "$context/Dockerfile"
cp "$root/deploy/requirements-container.txt" "$context/requirements.txt"
printf '\ntorch==2.8.0\ntorchvision==0.23.0\n' >> "$context/requirements.txt"
# Use a valid CA bundle, including your organization's root only if already trusted.
# Never disable TLS verification; CA files contain public certificates, not private keys.
cp "${SAR_BUILD_CA:-/etc/ssl/certs/ca-certificates.crt}" "$context/build-ca.pem"
DOCKER_BUILDKIT=0 docker build --platform linux/amd64 \
  --build-arg "PYTHON_BASE=$resolved" \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  -t "$image" "$context"
docker run --rm --network none "$image" python -c 'import torch; print(torch.__version__,torch.version.cuda)'
echo 'Environment built. Mount the separately supplied source using scripts/offline/run.sh.'
