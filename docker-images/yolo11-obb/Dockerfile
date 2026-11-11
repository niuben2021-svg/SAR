FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 MPLBACKEND=Agg YOLO_OFFLINE=true YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/opt/yolo-config NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY constraints.txt /opt/sar-yolo/constraints.txt
RUN python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu121 torchvision==0.18.1+cu121
COPY source/ultralytics /opt/ultralytics
RUN python -m pip install -c /opt/sar-yolo/constraints.txt setuptools==80.9.0 wheel \
    numpy==1.26.4 Pillow==10.4.0 matplotlib==3.8.4 scipy==1.13.1 pyyaml==6.0.2 /opt/ultralytics \
    && python -m pip check
COPY weights /workspace/weights
COPY scripts /opt/sar-yolo/scripts
RUN python /opt/sar-yolo/scripts/setup_runtime.py
WORKDIR /workspace
CMD ["bash"]
