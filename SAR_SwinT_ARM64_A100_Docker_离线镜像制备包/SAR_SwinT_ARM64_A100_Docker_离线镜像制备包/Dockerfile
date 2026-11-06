# The official NGC PyTorch 24.10 image supports ARM SBSA / CUDA 12.6.
# No RUN instructions: can cross-assemble the ARM64 filesystem from an x86_64
# Docker host, without emulating ARM binaries. GPU operations MUST be checked
# after import on the real ARM64 A100 server.
FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:24.10-py3
LABEL description="SAR Swin-T P1-P5 Faster R-CNN, ARM64 (Kunpeng + A100) offline" \
      org.opencontainers.image.version="1.0"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /workspace/sar
COPY project/ /workspace/sar/
COPY weights/ /workspace/weights/
CMD ["/bin/bash"]
