import os, sys
import torch, torchvision, mmcv, mmengine, mmdet
from mmcv.ops import MultiScaleDeformableAttention, nms
from mmdet.models.detectors.cvmt import CVMT
print('python:',sys.version.split()[0])
for name, obj in [('torch',torch),('torchvision',torchvision),('mmcv',mmcv),('mmengine',mmengine),('mmdet',mmdet)]:
    print(name, obj.__version__, obj.__file__)
assert mmdet.__file__.startswith('/opt/CVMT/'), 'imported wrong MMDetection'
print('CUDA:',torch.version.cuda, 'available:',torch.cuda.is_available())
if torch.cuda.is_available():
    x=torch.tensor([[0.,0.,10.,10.]],device='cuda')
    s=torch.tensor([.9],device='cuda')
    keep=nms(x,s,0.5)[1]
    print('CUDA nms OK:', keep.tolist(), 'device:',torch.cuda.get_device_name(0))
print('CVMT imports OK (not yet an end-to-end checkpoint inference test)')
