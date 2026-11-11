#!/usr/bin/env bash
set -euo pipefail
docker run --rm -it --gpus all --ipc=host --shm-size=12g \
    -v /root/data1:/root/data1 \
    --name cvmt_baseline cvmt:cu118 bash
