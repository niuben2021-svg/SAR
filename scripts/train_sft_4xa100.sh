#!/usr/bin/env bash
# MVP-05 validation entrypoint, intentionally never launches training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m sar_vlm.training.config --config configs/qwen3vl_lora_4xa100.yaml "$@"
