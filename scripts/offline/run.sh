#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo 'Usage: run.sh MODEL_DIR DATA_DIR OUTPUT_DIR COMMAND [ARGS...]'
  echo 'Commands: cpu/check/verify-model/forward/inspect-data/preview-complex/prepare-sft/train-smoke/train/help'
  exit 2
fi
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="$(realpath -e "$1")"
data_dir="$(realpath -e "$2")"
mkdir -p "$3"
output_dir="$(realpath -e "$3")"
shift 3
mode="$1"
# Only commands that need GPUs request NVIDIA runtime; CPU/transfer checks work without it.
gpu_args=()
case "$mode" in check|forward|train-smoke|train) gpu_args=(--gpus "${SAR_GPUS:-all}");; esac
if [[ "$mode" == shell && "${SAR_SHELL_GPU:-0}" == 1 ]]; then
  gpu_args=(--gpus "${SAR_GPUS:-all}")
fi
tty_args=()
if [[ "${SAR_INTERACTIVE:-0}" == 1 ]]; then tty_args=(-it); fi
# No mounts of host Docker socket, root filesystem, or privileged mode are needed.
docker run --rm --pull never --network none "${gpu_args[@]}" "${tty_args[@]}" \
  --env-file "$code_dir/deploy/offline.env" \
  --workdir /opt/sar-vlm --env PYTHONPATH=/opt/sar-vlm/src \
  --entrypoint /bin/bash \
  --mount "type=bind,source=$code_dir,target=/opt/sar-vlm,readonly" \
  --tmpfs "/tmp:rw,nosuid,nodev,size=${SAR_TMP_SIZE:-2g}" \
  --shm-size "${SAR_SHM_SIZE:-8g}" \
  --mount "type=bind,source=$model_dir,target=/models/qwen3vl4b,readonly" \
  --mount "type=bind,source=$data_dir,target=/data,readonly" \
  --mount "type=bind,source=$output_dir,target=/outputs" \
  "${SAR_IMAGE_TAG:-sar-vlm:qwen3vl4b}" /opt/sar-vlm/deploy/entrypoint.sh "$@"
