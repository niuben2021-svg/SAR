# Build on an internet-connected Linux x86_64 host that already has the old base image.
ARG BASE_IMAGE=mmrota-swin:latest
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    MMCV_WITH_OPS=1
COPY constraints.txt /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt
WORKDIR /opt
COPY CVMT /opt/CVMT
COPY verify_cvmt.py /opt/verify_cvmt.py
COPY wheelhouse/ /opt/wheelhouse/
# Do not update PyTorch. Remove only OpenMMLab 0.x stack in this NEW image.
RUN python -m pip uninstall -y mmrotate mmdet mmcv mmcv-full mmcv-lite mmengine || true
RUN python -m pip install --no-index --find-links=/opt/wheelhouse \
      'numpy==1.24.4' 'mmengine==0.10.5' 'mmcv==2.0.1' \
      'timm==0.9.16' 'scipy' 'shapely' 'pycocotools' 'terminaltables' 'six' 'tqdm' \
    && python -m pip install --no-deps --no-build-isolation /opt/CVMT \
    && cd /opt/CVMT && PYTHONPATH=/opt/CVMT python /opt/verify_cvmt.py
RUN python -m pip check
ENV PYTHONPATH=/opt/CVMT
COPY constraints.txt /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt
WORKDIR /opt/CVMT
CMD ["/bin/bash"]
