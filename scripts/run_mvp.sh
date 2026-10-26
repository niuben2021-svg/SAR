#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:---cpu}"
if [[ "$mode" != "--cpu" && "$mode" != "--forward" ]]; then
  echo 'Usage: scripts/run_mvp.sh [--cpu|--forward] (forward requires SAR_IMAGE and SAR_QUERY)'
  exit 1
fi
python -m sar_vlm.cli.check_env --config configs/mvp.yaml
python -m sar_vlm.cli.smoke_data --config configs/mvp.yaml
python -m pytest -q -m 'not gpu'
if [[ "$mode" == "--forward" ]]; then
  : "${SAR_IMAGE:?Set SAR_IMAGE to an authorized existing image}"
  : "${SAR_QUERY:?Set SAR_QUERY to the text query}"
  python -m sar_vlm.cli.smoke_forward --config configs/mvp.yaml --image "$SAR_IMAGE" --query "$SAR_QUERY"
else
  echo 'CPU data/tests complete. Real Qwen GPU forward NOT RUN by this command.'
fi
