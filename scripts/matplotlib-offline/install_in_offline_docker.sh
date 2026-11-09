#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON_FOR_INSTALL:-python}
"$PY" - <<'PY'
import sys,platform
if sys.version_info[:2] != (3,10):
    raise SystemExit(f'ERROR: requires Python 3.10, found {sys.version.split()[0]}')
if platform.system() != 'Linux' or platform.machine() not in {'x86_64','AMD64'}:
    raise SystemExit(f'ERROR: requires Linux x86_64, found {platform.system()} {platform.machine()}')
print('Verified:',sys.version.split()[0],platform.system(),platform.machine())
PY
(cd wheels && sha256sum -c ../SHA256SUMS)
"$PY" -m pip install \
  --no-index \
  --find-links="$(pwd)/wheels" \
  --constraint="$(pwd)/constraints.txt" \
  'matplotlib==3.8.4'
"$PY" - <<'PY'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
from pathlib import Path
plt.plot([0,1,2],[1,3,2]); plt.savefig('matplotlib_offline_test.png'); plt.close()
print('SUCCESS matplotlib:',matplotlib.__version__,'numpy:',numpy.__version__)
print('Test plot:',Path('matplotlib_offline_test.png').resolve())
PY
