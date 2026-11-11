"""Test actual pretrained OBB inference; synthetic data checks engineering only."""
import argparse
import gc
import platform
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision
import ultralytics
from ultralytics import YOLO
from torchvision.ops import nms

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--train-smoke', action='store_true')
args = parser.parse_args()
assert platform.machine() == 'x86_64'
assert ultralytics.__version__ == '8.4.209'
assert torch.__version__.split('+')[0] == '2.3.1'
assert torch.version.cuda == '12.1'
assert torchvision.__version__ == '0.18.1+cu121'
torch.set_num_threads(2)
device = 'cuda:0' if args.gpu else 'cpu'
if args.gpu:
    assert torch.cuda.is_available(), 'Check NVIDIA Container Toolkit and --gpus all'
    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
assert nms(boxes, torch.tensor([.9, .8], device=device), .5).tolist() == [0]
for size in ('n',) if args.gpu else 'nsmlx':
    model = YOLO(f'/workspace/weights/yolo11{size}-obb.pt')
    assert model.task == 'obb'
    results = model.predict(np.zeros((64, 64, 3), dtype=np.uint8), imgsz=64,
                            device=device, verbose=False, save=False)
    assert len(results) == 1 and results[0].obb is not None
    print(f'YOLO11{size} OBB pretrained loading and {device} inference: PASS')
    del model, results
    gc.collect()
if args.train_smoke:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for split in ('train', 'val'):
            (root / 'images' / split).mkdir(parents=True)
            (root / 'labels' / split).mkdir(parents=True)
            for index in range(2):
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                cv2.rectangle(image, (16, 16), (48, 48), (255, 255, 255), -1)
                cv2.imwrite(str(root / 'images' / split / f'{index}.png'), image)
                (root / 'labels' / split / f'{index}.txt').write_text('0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n')
        yaml = root / 'fixture.yaml'
        yaml.write_text(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: engineering_fixture\n')
        model = YOLO('/workspace/weights/yolo11n-obb.pt')
        model.train(data=str(yaml), epochs=1, imgsz=64, batch=2, workers=0, device='cpu',
                    optimizer='SGD', amp=False, plots=False, save=True, val=False,
                    project=str(root / 'runs'), name='smoke', exist_ok=True, mosaic=0, close_mosaic=0)
        assert (root / 'runs/smoke/weights/last.pt').is_file()
        print('One-epoch synthetic OBB training/backward/checkpoint: PASS; no SAR accuracy claim')
print('CHECK PASSED; real RTX4090 GPU validation must run on target server')
