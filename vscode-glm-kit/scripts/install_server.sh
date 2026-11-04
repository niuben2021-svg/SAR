#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then echo 'Usage: bash install_server.sh /path/to/server.tar.gz'; exit 2; fi
[[ "$(uname -m)" == x86_64 ]] || { echo 'This server package requires x86_64 Linux.' >&2; exit 1; }
commit=a717861c07586a53d83b80d742cd3f59b20a6273
expected=53a8e2707f27b3dfc08f8aef536f9595f4c479922fc13f9644e723b3b1abab37
archive="$(realpath -e "$1")"
printf '%s  %s\n' "$expected" "$archive" | sha256sum --check -
target="$HOME/.vscode-server/bin/$commit"
[[ ! -e "$target" ]] || { echo "Already exists; refusing to overwrite: $target"; exit 2; }
mkdir -p "$(dirname "$target")"
temporary="$(mktemp -d "$(dirname "$target")/.install-XXXXXXXX")"
trap 'rm -rf "$temporary"' EXIT
tar -xzf "$archive" -C "$temporary" --strip-components=1
"$temporary/node" --version
"$temporary/bin/code-server" --version
mv "$temporary" "$target"
echo "Installed $target. Match the Windows settings and VS Code version in README."
