#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
ASSETS="$ROOT/release-assets"
PACKAGE="$ROOT/SAR_SwinT_P1_RTX4090_CUDA121_Docker_Package"
docker save "$IMAGE" | gzip -1 | split -d --numeric-suffixes=1 -a 3 -b 1500000000 - "$ASSETS/$NAME.part"
cat "$ASSETS/$NAME".part* | gzip -t
cat "$ASSETS/$NAME".part* | sha256sum | sed "s/  -$/  $NAME/" > "$ASSETS/$NAME.sha256"
# Remove our built images/cache, then import the saved layers without a registry pull.
docker image rm "$IMAGE" "$BASE"
docker builder prune -af
cat "$ASSETS/$NAME".part* | gzip -dc | docker load 2>&1 | tee "$ASSETS/reimport-check.log"
[[ "$(docker image inspect "$IMAGE" --format '{{.Architecture}}')" == amd64 ]]
docker run --rm --network none "$IMAGE" python -c 'import torch,torchvision; from model import build_detector; assert torch.version.cuda == "12.1"; print("Reimported AMD64 CUDA121 offline runtime: PASS")' | tee -a "$ASSETS/reimport-check.log"
tar -czf "$ASSETS/SAR_SwinT_RTX4090_Import_Toolkit.tar.gz" -C "$PACKAGE" scripts README_先看我.md project/内网部署说明.md
cp scripts/release/rtx4090/import-rtx4090.sh scripts/release/rtx4090/IMPORT.md "$ASSETS/"
cp scripts/release/rtx4090/check_cpu.py "$ASSETS/"
cd "$ASSETS"
sha256sum ./* > "$ROOT/checksums.tmp"
mv "$ROOT/checksums.tmp" SHA256SUMS
