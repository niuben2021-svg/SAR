#!/usr/bin/env python3
import sys,platform,importlib.metadata
import numpy,scipy,cv2,h5py
from PIL import Image
from scipy.signal.windows import taylor
import torch,torchvision
from torchvision.models import swin_t,resnet50
from torchvision.ops import nms,MultiScaleRoIAlign
print('Python',sys.version.split()[0], 'OS',platform.machine())
for name in ['numpy','scipy','opencv-python-headless','Pillow','h5py','torch','torchvision']:
    try: print(name,importlib.metadata.version(name))
    except Exception as exc: print(name,'UNKNOWN:',exc)
if not torch.cuda.is_available():raise SystemExit('ERROR: CUDA not available')
print('CUDA available; gpu_count',torch.cuda.device_count())
print('GPU names:',[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,12.,12.]],device='cuda')
scores=torch.tensor([0.9,0.8],device='cuda')
print('TorchVision CUDA NMS:', nms(boxes,scores,.5).tolist())
print('OpenCV rotated IoU test:',cv2.rotatedRectangleIntersection(((0.,0.),(5.,5.),0.),((0.,0.),(5.,5.),0.))[0])
print('scipy Taylor:',taylor(8,4,30).shape)
print('OK: Core SSFG dependency imports and CUDA ops passed')
