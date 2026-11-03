#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(id -u)" == 0 ]] || { echo 'Run as root inside the container.' >&2; exit 1; }
[[ "$(dpkg --print-architecture)" == amd64 ]] || { echo 'Requires AMD64.' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == debian && "$VERSION_ID" == 12 ]] || { echo 'Requires Debian12.' >&2; exit 1; }
sha256sum --check SHA256SUMS
dpkg --install --refuse-downgrade debs/*.deb
unzip -v
echo 'Offline installation completed.'
