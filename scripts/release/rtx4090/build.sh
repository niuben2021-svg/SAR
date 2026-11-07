#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
PACKAGE="$ROOT/SAR_SwinT_P1_RTX4090_CUDA121_Docker_Package"
cd "$PACKAGE"
curl -fL --retry 4 --connect-timeout 30 --max-time 600 \
  https://download.pytorch.org/models/swin_t-704ceda3.pth -o weights/swin_t-704ceda3.pth
WEIGHT_SHA="$(sha256sum weights/swin_t-704ceda3.pth | cut -d ' ' -f1)"
[[ "$WEIGHT_SHA" == 704ceda3* ]] || { echo 'Official weight SHA256 prefix mismatch'; exit 1; }
docker pull --platform linux/amd64 "$BASE"
docker build --platform linux/amd64 --pull=false -t "$IMAGE" . 2>&1 | tee "$ROOT/release-assets/build.log"
[[ "$(docker image inspect "$IMAGE" --format '{{.Architecture}}')" == amd64 ]]
docker image inspect "$IMAGE" > "$ROOT/release-assets/image-inspect.json"
docker image inspect "$BASE" --format '{{json .RepoDigests}}' > "$ROOT/release-assets/base-digest.json"
printf 'Task ID: IMAGE-RTX4090-CU121\nSource commit: %s\nWeight SHA256: %s\nBuild platform: linux/amd64\nGPU validation and GPU peak memory: NOT RUN (runner has no RTX4090)\n' \
  "$GITHUB_SHA" "$WEIGHT_SHA" > "$ROOT/release-assets/BUILD-INFO.txt"
