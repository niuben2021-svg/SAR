"""Verify packaged runtime offline on CPU; synthetic inputs test engineering only."""
import platform

import numpy as np
import PIL
import torch
import torchvision
from torchvision.models import swin_t
from torchvision.ops import nms, roi_align

from model import build_detector

torch.set_num_threads(2)
torch.manual_seed(42)
assert platform.machine() == "x86_64"
assert torch.__version__.split("+")[0] == "2.3.1"
assert torch.version.cuda == "12.1"
assert torchvision.__version__ == "0.18.1+cu121"
assert np.__version__ == "1.26.4"
assert PIL.__version__ == "10.4.0"
print("VERSIONS:", platform.machine(), torch.__version__, torchvision.__version__, np.__version__, PIL.__version__)
boxes = torch.tensor([[0., 0., 20., 20.], [1., 1., 18., 18.], [25., 25., 35., 35.]])
assert nms(boxes, torch.tensor([.9, .6, .8]), .5).tolist() == [0, 2]
assert roi_align(torch.ones(1, 4, 64, 64), torch.tensor([[0., 4., 4., 24., 24.]]), (7, 7)).shape == (1, 4, 7, 7)
state = torch.load("/workspace/weights/swin_t-704ceda3.pth", map_location="cpu", weights_only=True)
swin_t(weights=None).load_state_dict(state, strict=True)
print("CPU NMS / ROIAlign / strict pretrained weight loading: PASS")
model = build_detector(2, image_size=64, swin_weights="/workspace/weights/swin_t-704ceda3.pth")
image = torch.rand(3, 64, 64)
model.train()
losses = model([image], [{"boxes": torch.tensor([[8., 8., 40., 40.]]), "labels": torch.tensor([1])}])
loss = sum(losses.values())
assert torch.isfinite(loss).item()
loss.backward()
assert model.backbone.p1_stem[0].weight.grad is not None
assert torch.isfinite(model.backbone.p1_stem[0].weight.grad).all().item()
model.eval()
with torch.inference_mode():
    result = model([image])[0]
assert {"boxes", "scores", "labels"} <= result.keys()
print("Swin-T + P1 detector CPU forward / backward / inference: PASS")
print("ALL CPU CHECKS PASSED; synthetic engineering input, no SAR accuracy claim; GPU checks NOT RUN")
