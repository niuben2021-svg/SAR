#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PYTHON_FOR_DOWNLOAD:-python3}
"$PY" -m pip --version >/dev/null || { echo 'ERROR: pip not available. Install python3-pip first.' >&2; exit 1; }
PACKAGE="matplotlib_py310_linux_x86_64_offline.tar.gz"
ROOT="matplotlib_py310_linux_x86_64_offline"
mkdir -p "$ROOT/wheels"
printf '\n== Download Matplotlib 3.8.4 and CPython 3.10 Linux x86_64 wheels ==\n'
"$PY" -m pip download \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --python-version 310 \
  --implementation cp \
  --abi cp310 \
  --constraint constraints.txt \
  --dest "$ROOT/wheels" \
  'matplotlib==3.8.4'
cp -f constraints.txt "$ROOT/constraints.txt"
cp -f install_in_offline_docker.sh "$ROOT/install.sh"
cp -f README.txt "$ROOT/README.txt"
chmod +x "$ROOT/install.sh"
# hashes for transfer verification
(cd "$ROOT/wheels"; sha256sum ./*.whl) > "$ROOT/SHA256SUMS"
COUNT=$(find "$ROOT/wheels" -maxdepth 1 -name '*.whl' | wc -l)
if [ "$COUNT" -lt 10 ]; then
  echo "ERROR: Only $COUNT wheels found, dependency download may be incomplete." >&2
  exit 1
fi
tar -czf "$PACKAGE" "$ROOT"
echo
printf 'SUCCESS: %s\n' "$(pwd)/$PACKAGE"
printf 'Wheel files: %s\n' "$COUNT"
ls -lh "$PACKAGE"
printf '\nUpload this tar.gz to your offline server, NOT this builder ZIP.\n'
