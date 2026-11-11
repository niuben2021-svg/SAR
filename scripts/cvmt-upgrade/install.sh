#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" - <<'PY'
import sys, platform, torch, torchvision
assert sys.version_info[:2] == (3,10) and platform.machine()=='x86_64', 'Requires Python3.10 Linux x86_64'
assert platform.system()=='Linux'
assert torch.__version__.startswith('2.0.1') and torch.version.cuda=='11.8', 'Existing torch2.0.1+cu118 required'
assert torchvision.__version__.startswith('0.15.2'), 'Existing torchvision0.15.2 required'
print('Existing GPU Torch versions verified; no torch/torchvision will be installed')
PY
sha256sum -c SHA256SUMS
# Deliberately replace the old OpenMMLab stack, as requested by the supplied Dockerfile.
"$PYTHON_BIN" -m pip uninstall -y mmrotate mmdet mmcv mmcv-full mmcv-lite mmengine
"$PYTHON_BIN" -m pip install --no-index --find-links wheels -c constraints.txt -r requirements.txt
"$PYTHON_BIN" -m pip install --no-index --no-deps --no-build-isolation ./CVMT
PYTHONPATH="$PWD/CVMT" "$PYTHON_BIN" verify_local.py
echo "Upgrade installed. For this shell run: export PYTHONPATH=\"$PWD/CVMT\""
