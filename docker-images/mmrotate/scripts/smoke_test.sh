#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mmrotate
python - <<'PY'
import torch, mmcv, mmdet, mmrotate
from mmcv.ops import nms_rotated, box_iou_rotated
from mmcv import Config
from mmdet.models import build_detector
c=Config.fromfile('configs/sar/swin_t_rotated_faster_rcnn_sar.py')
m=build_detector(c.model)
print('VERSIONS',torch.__version__,mmcv.__version__,mmdet.__version__,mmrotate.__version__)
print('MODEL BUILD PASS',type(m).__name__)
if torch.cuda.is_available():
    boxes=torch.tensor([[100.,100.,40.,20.,0.]],device='cuda')
    print('ROTATED IOU',box_iou_rotated(boxes,boxes).item())
    print('CUDA OPS PASS')
PY
