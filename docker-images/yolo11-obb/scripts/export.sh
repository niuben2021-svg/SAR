#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
ASSETS="$ROOT/release-assets"
NAME=yolo11_obb_cuda12.1_amd64.tar.gz
PACKAGE="$ROOT/docker-images/yolo11-obb"
docker image inspect "$IMAGE" > "$ASSETS/image-inspect.json"
docker image inspect "$BASE" --format '{{json .RepoDigests}}' > "$ASSETS/base-digest.json"
docker run --rm --network none "$IMAGE" python -m pip freeze > "$ASSETS/requirements-actual.txt"
cp "$PACKAGE/weights/SHA256SUMS" "$ASSETS/weights-SHA256SUMS.txt"
git -C "$PACKAGE/source/ultralytics" archive --format=tar.gz --prefix=ultralytics-8.4.209/ HEAD > "$ASSETS/ultralytics-8.4.209-source.tar.gz"
docker save "$IMAGE" | gzip -1 | split -d --numeric-suffixes=1 -a 3 -b 1500000000 - "$ASSETS/$NAME.part"
cat "$ASSETS/$NAME".part* | gzip -t
cat "$ASSETS/$NAME".part* | sha256sum | sed "s/  -$/  $NAME/" > "$ASSETS/$NAME.sha256"
docker image rm "$IMAGE" "$BASE"
docker builder prune -af
cat "$ASSETS/$NAME".part* | gzip -dc | docker load 2>&1 | tee "$ASSETS/reimport.log"
docker run --rm --network none "$IMAGE" python -c 'from ultralytics import YOLO; m=YOLO("/workspace/weights/yolo11n-obb.pt"); assert m.task=="obb"; print("Reimported offline pretrained model: PASS")' | tee -a "$ASSETS/reimport.log"
cp "$PACKAGE/IMPORT.md" "$PACKAGE/scripts/03_offline_load_and_run.sh" "$ASSETS/"
tar -czf "$ASSETS/yolo11-build-source.tar.gz" -C "$ROOT" .github/workflows/yolo11-obb-release.yml \
  docker-images/yolo11-obb/Dockerfile docker-images/yolo11-obb/constraints.txt docker-images/yolo11-obb/scripts docker-images/yolo11-obb/IMPORT.md
printf 'Task ID: IMAGE-YOLO11-OBB\nStatus: DONE (image build and offline CPU verification)\nChanged files: docker-images/yolo11-obb; .github/workflows/yolo11-obb-release.yml\nCommands executed: docker build; pip check; check_image.py --train-smoke; docker save; docker load\nTests: dependency check, five OBB CPU inferences, synthetic one-epoch training and reimport PASSED; GPU NOT RUN\nArtifacts: image parts, source, hashes, logs\nActual hardware/software: GitHub x86_64 CPU runner; CUDA12.1 runtime; exact pip versions in requirements-actual.txt\nMeasured GPU peak memory: NOT RUN\nOpen risks/next dependency: internal RTX4090 CUDA and four-GPU checks\nBuild source commit: %s\nUltralytics commit: 9b57ed7e6fd366a95047aaa0526f542a62b202f1\n' "$GITHUB_SHA" > "$ASSETS/BUILD-INFO.txt"
cd "$ASSETS"
sha256sum ./* > "$ROOT/yolo-checksums.tmp"
mv "$ROOT/yolo-checksums.tmp" SHA256SUMS
