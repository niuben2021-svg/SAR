#!/usr/bin/env bash
set -euo pipefail
echo '=== Host ==='
uname -m
cat /etc/os-release | head -5 || true
echo '=== Driver and GPUs ==='
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo '=== Docker ==='
docker version
echo '=== Docker runtimes ==='
docker info --format '{{json .Runtimes}}'
echo '=== Docker disk ==='
df -h /var/lib/docker || true
echo '=== GPU runtime prerequisites ==='
if command -v nvidia-container-runtime >/dev/null 2>&1; then
  echo "nvidia-container-runtime: $(command -v nvidia-container-runtime)"
else
  echo 'WARNING: NVIDIA Container Toolkit runtime not found in PATH. --gpus all might fail.'
fi
echo 'Server checks complete (GPU passthrough not yet verified).'
