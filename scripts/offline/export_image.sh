#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then echo 'Usage: export_image.sh IMAGE OUTPUT.tar.gz'; exit 2; fi
image="$1"
output="$2"
if [[ -e "$output" || -e "$output.sha256" ]]; then echo 'Refusing to overwrite export' >&2; exit 2; fi
mkdir -p "$(dirname "$output")"
# pipefail preserves failures in docker save, including full destination disks.
partial="$(mktemp "${output}.partial.XXXXXX")"
trap 'rm -f "$partial"' EXIT
docker save "$image" | gzip -1 > "$partial"
mv "$partial" "$output"
(
  cd "$(dirname "$output")"
  sha256sum "$(basename "$output")" > "$(basename "$output").sha256"
)
echo "Exported $output and checksum. Weights and datasets are separate mounts."
