"""Test CUDA ARM64 and GPU torchvision.ops, without pulling dependencies."""
import argparse
import platform

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--require-cuda', action='store_true')
    args = p.parse_args()
    import numpy, PIL, torch, torchvision
    from torchvision.ops import nms, roi_align
    from torchvision.models import swin_t
    print('ARCH:', platform.machine())
    print('Python:', platform.python_version())
    print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
    print('TorchVision:', torchvision.__version__)
    print('NumPy:', numpy.__version__, 'Pillow:', PIL.__version__)
    print('CUDA:', torch.cuda.is_available(), 'GPU count:', torch.cuda.device_count())
    print('TorchVision Swin-T:', swin_t(weights=None).__class__.__name__)
    cpu_boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    print('TorchVision CPU NMS:', nms(cpu_boxes, scores, .5).tolist())
    if args.require_cuda:
        assert platform.machine() in ('aarch64','arm64'), 'Wrong CPU architecture; need ARM64 image.'
        assert torch.cuda.is_available(), 'CUDA not exposed: check nvidia-container-toolkit/--gpus all.'
        assert torch.cuda.device_count() >= 2, 'Expected two A100 cards.'
        for i in range(torch.cuda.device_count()):
            print('GPU', i, torch.cuda.get_device_name(i))
        cb = cpu_boxes.to('cuda'); sc = scores.to('cuda')
        print('TorchVision CUDA NMS:', nms(cb, sc, .5).cpu().tolist())
        x = torch.arange(256,device='cuda',dtype=torch.float32).reshape(1,1,16,16)
        roi = torch.tensor([[0.,0.,0.,14.,14.]],device='cuda')
        result=roi_align(x,roi,(4,4),spatial_scale=1.)
        print('TorchVision CUDA ROIAlign:',tuple(result.shape))
        w = torch.randn(64,64,device='cuda'); print('CUDA matmul norm:',float((w@w).norm()))
    print('CHECK PASSED')
if __name__ == '__main__': main()
