import torch, torchvision, mmcv, mmengine, mmdet
from pathlib import Path
from mmcv.ops import MultiScaleDeformableAttention, nms
from mmdet.models.detectors.cvmt import CVMT
assert Path(mmdet.__file__).resolve().is_relative_to(Path(__file__).parent.resolve()/'CVMT')
assert mmcv.__version__=='2.0.1' and mmengine.__version__=='0.10.5'
print('CVMT upgrade imports PASS:', torch.__version__, torchvision.__version__, mmcv.__version__, mmdet.__file__)
if torch.cuda.is_available():
    boxes=torch.tensor([[0.,0.,10.,10.]],device='cuda')
    assert nms(boxes, torch.tensor([.9],device='cuda'), .5)[1].tolist()==[0]
    print('CUDA NMS PASS')
else:
    print('GPU NOT AVAILABLE; CUDA execution not verified')
