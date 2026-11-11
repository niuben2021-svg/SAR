#!/usr/bin/env bash
set -euo pipefail
BASE_IMAGE=${BASE_IMAGE:-mmrota-swin:latest}
cd "$(dirname "$0")"
test -d CVMT || { echo 'CVMT source missing (copy official CVMT checkout here)'; exit 1; }
ls wheelhouse/mmcv-2.0.1-*.whl >/dev/null || { echo 'mmcv binary wheel missing'; exit 1; }
docker build --network none --build-arg BASE_IMAGE="$BASE_IMAGE" -t cvmt:cu118 -f Dockerfile .
