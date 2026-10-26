#!/usr/bin/env bash
set -euo pipefail
cd /opt/sar-vlm
mode="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  help)
    cat <<'HELP'
SAR-VLM offline container commands:
  cpu                 CPU data smoke + tests; no model directory required
  verify-model        Verify /models/qwen3vl4b transfer hashes and required files
  check               GPU + local model layout preflight (add --no-gpu for CPU inspection)
  forward --image /data/preview.png --query TEXT
  inspect-data --root /data/FAIR-CSAR --output /outputs/fair_csar_inventory.json
  preview-complex --input /data/scene.npy --output /outputs/scene.png [--key NAME]
  prepare-sft --csv /data/labels.csv --image-root /data/images --output /outputs/sft ...
  train --dataset /outputs/sft/train.jsonl --review /outputs/sft/train.review.json --smoke-report PATH [--execute]
  train-smoke --dataset /data/reviewed_sft.jsonl --review /data/review.json [--execute]
  download-model --source modelscope --destination /downloads/qwen3vl4b
  shell               Start an interactive bash shell
Use --network none for inner-network validation. Weights/data are external mounts.
HELP
    ;;
  cpu)
    python -m sar_vlm.cli.smoke_data --config configs/cpu.yaml
    exec python -m pytest -q -o cache_dir=/outputs/pytest-cache -m 'not gpu' "$@"
    ;;
  verify-model) exec python -m sar_vlm.deployment.artifacts verify /models/qwen3vl4b "$@" ;;
  check) exec python -m sar_vlm.deployment.preflight "$@" ;;
  forward)
    exec python -m sar_vlm.cli.smoke_forward --config configs/offline.yaml "$@"
    ;;
  inspect-data) exec python -m sar_vlm.cli.inspect_dataset "$@" ;;
  preview-complex) exec python -m sar_vlm.cli.prepare_complex_preview "$@" ;;
  prepare-sft) exec python -m sar_vlm.cli.prepare_sft "$@" ;;
  train) exec python -m sar_vlm.training.train "$@" ;;
  train-smoke) exec python -m sar_vlm.training.offline_smoke "$@" ;;
  download-model)
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
    exec python -m sar_vlm.deployment.download_model "$@"
    ;;
  shell) exec bash "$@" ;;
  *) echo "Unknown command: $mode (use help)" >&2; exit 2 ;;
esac
