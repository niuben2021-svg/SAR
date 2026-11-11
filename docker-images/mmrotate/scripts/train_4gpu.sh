#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mmrotate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-12}"
export SAR_DATA_ROOT="${SAR_DATA_ROOT:-/data/sar}"
export SAR_CLASSES="${SAR_CLASSES:-airplane}"
mkdir -p /workspace/work_dirs/sar_swin_t_rfrcnn
bash tools/dist_train.sh configs/sar/swin_t_rotated_faster_rcnn_sar.py 4 \
  --work-dir /workspace/work_dirs/sar_swin_t_rfrcnn "$@"
