"""Validate host GPU visibility, PyTorch/TorchVision CUDA extension operators and model training forward."""
import argparse
import platform
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-gpus', type=int, default=0)
    ap.add_argument('--run-model', action='store_true', help='also verify model train forward on GPU using random data')
    args = ap.parse_args()
    import numpy as np
    import torch
    import torchvision
    from torchvision.ops import nms, roi_align

    print('platform:', platform.platform(), 'arch:', platform.machine())
    print('python:', sys.version.split()[0])
    print('torch:', torch.__version__, 'torch.cuda:', torch.version.cuda)
    print('torchvision:', torchvision.__version__, 'numpy:', np.__version__)
    print('CUDA available:', torch.cuda.is_available())
    count = torch.cuda.device_count()
    print('Visible GPU count:', count)
    if args.require_gpus and count != args.require_gpus:
        raise RuntimeError(f'Expected {args.require_gpus} CUDA devices, got {count}. Check NVIDIA Container Toolkit and Docker GPU flags.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available inside container. Check --gpus all and NVIDIA Container Toolkit.')
    if torch.version.cuda is None or not torch.version.cuda.startswith('12.1'):
        raise RuntimeError(f'Expected torch CUDA 12.1, got {torch.version.cuda}')
    for i in range(count):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    device='cuda:0'
    x=torch.rand(128,128,device=device)
    assert torch.isfinite(x @ x).all().item()
    boxes=torch.tensor([[0,0,20,20],[1,1,18,18],[25,25,35,35]],dtype=torch.float32,device=device)
    scores=torch.tensor([0.9,0.6,0.8],device=device)
    selected=nms(boxes,scores,0.5)
    assert selected.numel()==2, f'NMS result wrong: {selected}'
    fmap=torch.randn(1,4,64,64,device=device)
    rois=torch.tensor([[0,4,4,24,24]],dtype=torch.float32,device=device)
    out=roi_align(fmap,rois,output_size=(7,7),spatial_scale=1.0,aligned=True)
    assert out.shape==(1,4,7,7),out.shape
    print('CUDA NMS + ROIAlign: PASS')
    if args.run_model:
        from model import build_detector
        model=build_detector(num_classes=2,image_size=128,use_p1=True).to(device)
        x=torch.randn(3,128,128,device=device)
        targets=[{'boxes':torch.tensor([[12.,12.,45.,50.]],device=device),
                  'labels':torch.tensor([1],dtype=torch.int64,device=device)}]
        model.train()
        losses=model([x],targets)
        loss=sum(losses.values())
        assert torch.isfinite(loss).item(), losses
        loss.backward()
        model.eval()
        with torch.no_grad():
            detections=model([x])[0]
        print('P1 Swin-T FasterRCNN train-forward/backward/inference: PASS; prediction count:',len(detections['boxes']))
    print('ALL CHECKS PASSED')


if __name__=='__main__':
    main()
