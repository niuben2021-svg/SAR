#!/usr/bin/env bash
# Use on an internet-connected Linux x86_64 machine with Docker installed and BASE image available.
set -euo pipefail
BASE_IMAGE=${BASE_IMAGE:-mmrota-swin:latest}
CVMT_BRANCH=${CVMT_BRANCH:-main}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
if [ ! -d CVMT/.git ]; then
  git clone --depth 1 --branch "$CVMT_BRANCH" https://github.com/RSIP-NJUPT/CVMT.git CVMT
fi
# Download dependencies using the exact base Python/Torch platform via a temporary container.
mkdir -p wheelhouse
cat > build_wheels_in_container.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import torch,sys
print('base python:',sys.version)
print('base torch:',torch.__version__,'cuda:',torch.version.cuda)
assert sys.version_info[:2]==(3,10), 'expected Python 3.10 base'
assert torch.__version__.startswith('2.0.1'), 'expected torch 2.0.1 base'
PY
export PIP_CONSTRAINT=/build/constraints.txt
python -m pip install -U 'pip<25' 'setuptools<75' wheel
# Resolve full transitive closure into the wheelhouse, including torch requirements (large). 
# We only install the incremental dependencies into the NEW final image.
python -m pip download --only-binary=:all: --dest /build/wheelhouse \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  'torch==2.0.1' 'torchvision==0.15.2' \
  'numpy==1.24.4' 'mmengine==0.10.5' 'timm==0.9.16' \
  'scipy' 'shapely' 'pycocotools' 'matplotlib' 'opencv-python-headless<4.12' \
  'Pillow' 'yapf<0.41' 'addict' 'rich' 'terminaltables' 'six' 'tqdm'
# Must retrieve the CUDA-op build of mmcv (NOT mmcv-lite or its source archive).
python -m pip download --no-deps --only-binary=:all: --dest /build/wheelhouse \
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
  'mmcv==2.0.1'
python - <<'PY'
from pathlib import Path
import sys
xs=list(Path('/build/wheelhouse').glob('mmcv-2.0.1-*.whl'))
if not xs: sys.exit('ERROR: binary mmcv 2.0.1 wheel not found; DO NOT silently use source/mmcv-lite')
print('MMCV wheel:',xs)
PY
INNER
chmod +x build_wheels_in_container.sh
docker run --rm --network host -v "$ROOT:/build" -w /build "$BASE_IMAGE" bash /build/build_wheels_in_container.sh
# preserve the original base; build a NEW derivative image using the offline local wheelhouse.
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t cvmt:cu118 -f Dockerfile .
echo 'Built image: cvmt:cu118'
echo 'Run: docker save -o cvmt_cu118.tar cvmt:cu118'
