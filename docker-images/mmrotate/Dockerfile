FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
COPY constraints.txt /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ca-certificates libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && ln -sf /usr/bin/pip3 /usr/local/bin/pip
RUN python -m pip install --upgrade 'pip<25' setuptools wheel
RUN python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install 'numpy==1.24.4' 'opencv-python-headless==4.8.1.78' 'Pillow<11' 'setuptools<70' 'packaging<25' 'yapf==0.40.1'
# Exact prebuilt MMCV CUDA ops: cu118 / torch2.0 / cp310; no compilation on internal machine.
RUN python -m pip install 'mmcv-full==1.7.2' -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html --only-binary mmcv-full
RUN python -m pip install 'mmdet==2.28.2' 'mmrotate==0.3.4' 'timm==0.9.12' 'terminaltables' 'pycocotools' 'shapely<2' 'tensorboard' 'einops'
RUN python -m pip install 'numpy==1.24.4' && python -m pip check
WORKDIR /workspace/mmrotate
RUN git clone --depth 1 --branch v0.3.4 https://github.com/open-mmlab/mmrotate.git /workspace/mmrotate
COPY configs /workspace/mmrotate/configs/sar
COPY scripts /workspace/scripts
RUN chmod +x /workspace/scripts/*.sh
RUN python -c 'import torch,mmcv,mmdet,mmrotate; from mmcv.ops import nms_rotated,box_iou_rotated; print("VERSIONS",torch.__version__,mmcv.__version__,mmdet.__version__,mmrotate.__version__); print("ROTATED OPS IMPORT OK")'
ENV PYTHONPATH=/workspace/mmrotate:$PYTHONPATH
CMD ["bash"]
