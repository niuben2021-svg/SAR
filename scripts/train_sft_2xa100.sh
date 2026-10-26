#!/usr/bin/env bash
# Validate the actual two-A100 settings; offline train-smoke is the bounded launcher.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m sar_vlm.training.config --config configs/qwen3vl_lora_2xa100.yaml "$@"
