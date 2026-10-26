#!/usr/bin/env bash
# Read-only host diagnostics; never installs packages or changes drivers/runtime configuration.
set -u
output="${1:-kylin-host-report.txt}"
if [[ -e "$output" ]]; then echo 'Refusing to overwrite an existing host report' >&2; exit 2; fi
{
  date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
  uname -m
  cat /etc/os-release
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    nvidia-smi topo -m
  else
    echo 'BLOCKED: nvidia-smi is unavailable'
  fi
  if command -v docker >/dev/null 2>&1; then
    docker version
    docker info --format 'storage={{.Driver}} runtimes={{json .Runtimes}}'
  else
    echo 'BLOCKED: Docker is unavailable'
  fi
  if command -v nvidia-ctk >/dev/null 2>&1; then nvidia-ctk --version; fi
  if command -v rpm >/dev/null 2>&1; then
    rpm -q nvidia-container-toolkit libnvidia-container1
  fi
  df -h "$(dirname "$output")"
  echo 'A real docker run --gpus request is still required; this report alone is not GPU readiness.'
} > "$output" 2>&1
cat "$output"
