#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
PACKAGE="$ROOT/SAR_SwinT_ARM64_A100_Docker_离线镜像制备包/SAR_SwinT_ARM64_A100_Docker_离线镜像制备包"
ASSETS="$ROOT/release-assets"
mkdir -p "$ASSETS" "$PACKAGE/weights"
cd "$PACKAGE"
curl -fL --retry 5 https://download.pytorch.org/models/swin_t-704ceda3.pth -o weights/swin_t-704ceda3.pth
WEIGHT_SHA="$(sha256sum weights/swin_t-704ceda3.pth | cut -d ' ' -f1)"
[[ "$WEIGHT_SHA" == 704ceda3* ]] || { echo 'Official weight hash prefix mismatch'; exit 1; }
docker pull --platform linux/arm64 nvcr.io/nvidia/pytorch:24.10-py3
docker build --platform linux/arm64 --pull=false -t "$IMAGE" .
[[ "$(docker image inspect "$IMAGE" --format '{{.Architecture}}')" == arm64 ]]
docker image inspect "$IMAGE" > "$ASSETS/image-inspect.json"
docker image inspect nvcr.io/nvidia/pytorch:24.10-py3 --format '{{json .RepoDigests}}' > "$ASSETS/base-digest.json"
docker run --rm --network none "$IMAGE" python check_env.py 2>&1 | tee "$ASSETS/arm64-cpu-check.log"
docker run --rm --network none --entrypoint python "$IMAGE" -c '
import torch
from torchvision.models import swin_t
from torchvision.ops import roi_align
from model import build_detector
torch.set_num_threads(2)
weights=torch.load("/workspace/weights/swin_t-704ceda3.pth",map_location="cpu",weights_only=True)
swin_t(weights=None).load_state_dict(weights,strict=True)
assert roi_align(torch.ones(1,1,16,16),torch.tensor([[0.,0.,0.,14.,14.]]),(4,4)).shape == (1,1,4,4)
model=build_detector(2,image_size=64,swin_weights="/workspace/weights/swin_t-704ceda3.pth")
model.eval()
with torch.inference_mode():
    result=model([torch.zeros(3,64,64)])[0]
assert {"boxes","scores","labels"} <= result.keys()
print("ARM64 offline weight loading, ROIAlign and detector forward: PASSED (engineering fixture only; no accuracy claim)")
' 2>&1 | tee "$ASSETS/offline-model-check.log"
docker run --rm --network none --entrypoint python "$IMAGE" -m pip freeze > "$ASSETS/requirements-actual.txt"
printf 'Source commit: %s\nWeight SHA256: %s\nGPU validation: NOT RUN; requires internal ARM64 A100 server\n' "$GITHUB_SHA" "$WEIGHT_SHA" > "$ASSETS/BUILD-INFO.txt"
NAME=SAR_SwinT_ARM64_A100_CUDA126_Image.tar.gz
docker save "$IMAGE" | gzip -1 | split -d --numeric-suffixes=1 -a 3 -b 1500000000 - "$ASSETS/$NAME.part"
cd "$ASSETS"
cat "$NAME".part* | gzip -t
cat "$NAME".part* | sha256sum | sed "s/  -$/  $NAME/" > "$NAME.sha256"
tar -czf SAR_SwinT_Import_Toolkit.tar.gz -C "$PACKAGE" scripts project/内网部署说明.md
cp "$ROOT/scripts/release/import-swin.sh" import-swin.sh
cp "$ROOT/scripts/release/IMPORT.md" IMPORT.md
sha256sum ./* > "$ROOT/work-checksums"
mv "$ROOT/work-checksums" SHA256SUMS
