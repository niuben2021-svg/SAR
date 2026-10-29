#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ "$(id -u)" == 0 ]] || { echo 'Run as root inside the container.' >&2; exit 1; }
[[ "$(dpkg --print-architecture)" == arm64 ]] || { echo 'Requires Debian ARM64 container.' >&2; exit 1; }
source /etc/os-release
[[ "$ID" == debian && "$VERSION_ID" == 12 ]] || { echo 'Requires Debian 12 container.' >&2; exit 1; }
sha256sum --check SHA256SUMS
dpkg --compare-versions "$(dpkg-query -W -f='${Version}' libc6)" ge 2.36 || {
  echo 'Installed libc6 must be >=2.36; this package does not replace libc6.' >&2; exit 1;
}
# No apt/network and no replacement of the host driver or base C library.
dpkg --install --refuse-downgrade debs/*.deb
dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' unrar
# Some versions return 0 and others return a usage code when no arguments are supplied.
set +e
unrar > /tmp/sar-unrar-help.txt 2>&1
result=$?
set -e
head -n 4 /tmp/sar-unrar-help.txt
if [[ "$result" != 0 && "$result" != 7 ]]; then exit "$result"; fi
echo 'Offline installation completed.' 
