#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" - <<'PY'
import sys, platform, importlib.metadata, hashlib
from pathlib import Path
if sys.version_info[:2] != (3, 11) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise SystemExit('Requires CPython 3.11 / Linux x86_64. Use your target training interpreter.')
if platform.python_implementation() != 'CPython':
    raise SystemExit('CPython is required')
for name in ('opencv-python', 'opencv-contrib-python', 'opencv-contrib-python-headless'):
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        continue
    raise SystemExit(f'Found {name}=={version}, which shares cv2. Resolve this conflict before installing headless OpenCV; nothing was uninstalled.')
for line in Path('SHA256SUMS').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = Path(name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('SHA256 verification failed: ' + name)
print('Python/architecture and wheel SHA256 checks passed')
PY
"$PYTHON_BIN" -m pip install --no-index --find-links wheels -r requirements.txt
"$PYTHON_BIN" check_cv2.py
