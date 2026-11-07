#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TAG="${IMAGE_TAG:-sar-swin-p1:rtx4090-cu121}"
OUT="${OUT_FILE:-$ROOT/dist/SAR_SwinT_RTX4090_CUDA121_Image.tar.gz}"
mkdir -p "$(dirname "$OUT")" weights
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: build machine is $(uname -m), expected x86_64" >&2; exit 1
fi
command -v docker >/dev/null || { echo 'ERROR: docker not installed' >&2; exit 1; }
if [[ ! -s weights/swin_t-704ceda3.pth ]]; then
  echo '[1/5] Downloading official TorchVision Swin-T weight...'
  if command -v curl >/dev/null; then
    curl --fail --location --retry 4 --connect-timeout 30 \
      -o weights/swin_t-704ceda3.pth.part \
      https://download.pytorch.org/models/swin_t-704ceda3.pth
  else
    wget -O weights/swin_t-704ceda3.pth.part https://download.pytorch.org/models/swin_t-704ceda3.pth
  fi
  mv weights/swin_t-704ceda3.pth.part weights/swin_t-704ceda3.pth
fi
ls -lh weights/swin_t-704ceda3.pth
printf '%s\n' '[2/5] Pulling official PyTorch CUDA 12.1 image...'
docker pull --platform linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
printf '%s\n' '[3/5] Building full x86_64 project image (network required)...'
docker build --platform linux/amd64 -t "$TAG" . 2>&1 | tee dist/build.log
ARCH="$(docker image inspect "$TAG" --format '{{.Architecture}}')"
[[ "$ARCH" == amd64 ]] || { echo "ERROR: image arch is $ARCH"; exit 1; }
printf '%s\n' '[4/5] Exporting real docker image tar.gz...'
docker save "$TAG" | gzip -1 > "$OUT"
[[ -s "$OUT" ]] || { echo 'ERROR: image archive is empty'; exit 1; }
printf '%s\n' '[5/5] SHA256 and size...'
(cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
ls -lh "$OUT" "$OUT.sha256"
echo "SUCCESS: actual Docker image (not source ZIP): $OUT"
