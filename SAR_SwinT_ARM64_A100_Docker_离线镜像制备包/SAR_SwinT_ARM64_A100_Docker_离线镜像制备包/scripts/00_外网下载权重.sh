#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
url="https://download.pytorch.org/models/swin_t-704ceda3.pth"
file="weights/swin_t-704ceda3.pth"
if [ -f "$file" ] && [ "$(stat -c %s "$file")" -gt 50000000 ]; then
 echo "Weights already present: $file"
else
 echo "Downloading official TorchVision Swin-T weights ..."
 curl -fL --retry 3 --connect-timeout 20 "$url" -o "$file.tmp"
 mv "$file.tmp" "$file"
fi
sha256sum "$file"
ls -lh "$file"
