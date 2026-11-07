"""Swin-T + true stride-2 P1 + P2..P5 FPN + Faster R-CNN."""
from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn
from torchvision.models import swin_t
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator


class SwinTinyP1FPN(nn.Module):
    """P1 is built directly from input at stride 2, fused top-down with Swin outputs."""

    out_channels = 128

    def __init__(self, weights_path=None, use_p1=True):
        super().__init__()
        self.swin = swin_t(weights=None)
        self.use_p1 = use_p1
        if weights_path:
            obj = torch.load(str(weights_path), map_location="cpu", weights_only=False)
            state = obj.get("state_dict", obj.get("model", obj)) if isinstance(obj, dict) else obj
            if any(k.startswith("module.") for k in state):
                state = {k.removeprefix("module."): v for k,v in state.items()}
            missing, unexpected = self.swin.load_state_dict(state, strict=False)
            if len(missing) > 10 or len(unexpected) > 10:
                raise RuntimeError(f"Checkpoint incompatible with torchvision swin_t. Missing={missing[:10]}; extra={unexpected[:10]}")
            print(f"Loaded Swin-T weights: {weights_path}; missing {len(missing)}; extra {len(unexpected)}")
        # 1/2-resolution high-frequency information is genuinely derived from input.
        self.p1_stem = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        ) if use_p1 else None
        channels = [48, 96, 192, 384, 768] if use_p1 else [96, 192, 384, 768]
        self.fpn = FeaturePyramidNetwork(channels, out_channels=self.out_channels)

    def forward(self, x):
        feature_maps = OrderedDict()
        if self.use_p1:
            feature_maps["p1"] = self.p1_stem(x)
        z = x
        stages = {1: "p2", 3: "p3", 5: "p4", 7: "p5"}
        for idx, layer in enumerate(self.swin.features):
            z = layer(z)  # NHWC
            if idx in stages:
                feature_maps[stages[idx]] = z.permute(0, 3, 1, 2).contiguous()
        return self.fpn(feature_maps)


def build_detector(num_classes, image_size=512, swin_weights=None, use_p1=True):
    """num_classes INCLUDES background class 0."""
    backbone = SwinTinyP1FPN(weights_path=swin_weights, use_p1=use_p1)
    levels = ["p1", "p2", "p3", "p4", "p5"] if use_p1 else ["p2", "p3", "p4", "p5"]
    sizes = (8, 16, 32, 64, 128) if use_p1 else (16, 32, 64, 128)
    # Same number of anchors at all FPN levels is required by the shared RPN head.
    anchor_generator = AnchorGenerator(
        sizes=tuple((s,) for s in sizes),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(levels),
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=levels,
        output_size=7,
        sampling_ratio=2,
        canonical_scale=224,
        canonical_level=4,
    )
    # Inputs are already cropped to exact native image_size. No silent 800-px resizing.
    return FasterRCNN(
        backbone, num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_size, max_size=image_size,
        image_mean=[0.485] * 3, image_std=[0.229] * 3,
        rpn_pre_nms_top_n_train=1200, rpn_post_nms_top_n_train=600,
        rpn_pre_nms_top_n_test=650, rpn_post_nms_top_n_test=350,
        box_detections_per_img=200, box_score_thresh=0.02,
    )
