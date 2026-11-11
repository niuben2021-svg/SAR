"""Offline engineering checks, without claiming GPU or SAR accuracy."""
import torch
import mmrotate  # register rotated detector/backbone components before build_detector
from mmcv import Config
from mmcv.runner import build_optimizer
from mmcv.ops import box_iou_rotated, nms_rotated
from mmdet.models import build_detector

torch.set_num_threads(2)
c = Config.fromfile('configs/sar/swin_t_rotated_faster_rcnn_sar.py')
m = build_detector(c.model)
m.init_weights()
o = build_optimizer(m, c.optimizer)
assert isinstance(o, torch.optim.AdamW)
assert m.roi_head.bbox_head.num_classes == len(c.classes)
boxes = torch.tensor([[100., 100., 40., 20., 0.], [100., 100., 40., 20., 0.]])
ious = box_iou_rotated(boxes, boxes)
assert torch.allclose(ious, torch.ones(2, 2), atol=1e-5)
_, indices = nms_rotated(boxes, torch.tensor([.9, .8]), .5)
assert len(indices) == 1
print('CPU rotated IoU/NMS, config, initialized Swin-T detector and AdamW: PASS')
print('GPU smoke and real training: NOT RUN; execute smoke_test.sh on internal server with --gpus all')
