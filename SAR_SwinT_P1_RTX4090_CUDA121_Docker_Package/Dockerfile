# Official PyTorch x86_64 image. CUDA 12.1 is supported by host NVIDIA driver 535.146.02.
FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace/project
# Prevent numpy 2.x ABI changes; install CUDA 12.1 torchvision wheel matching torch 2.3.1.
RUN python -m pip install --no-cache-dir "numpy==1.26.4" "Pillow==10.4.0" \
    && python -m pip install --no-cache-dir --no-deps \
       --index-url https://download.pytorch.org/whl/cu121 \
       "torchvision==0.18.1+cu121" \
    && python -c "import torch,torchvision,numpy,PIL; assert torch.__version__.startswith('2.3.1'), torch.__version__; assert torch.version.cuda.startswith('12.1'), torch.version.cuda; assert torchvision.__version__.startswith('0.18.1'), torchvision.__version__; print('Torch/Vision import versions: PASS', torch.__version__, torchvision.__version__)"

COPY project/ /workspace/project/
COPY weights/swin_t-704ceda3.pth /workspace/weights/swin_t-704ceda3.pth
# Check pretrained checkpoint key compatibility during build, no GPU necessary.
RUN python -c "import torch; from torchvision.models import swin_t; d=torch.load('/workspace/weights/swin_t-704ceda3.pth',map_location='cpu'); m=swin_t(weights=None); m.load_state_dict(d,strict=True); print('Swin-T checkpoint load: PASS')"

CMD ["bash"]
