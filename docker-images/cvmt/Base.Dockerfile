FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ca-certificates libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/local/bin/python
RUN python -m pip install 'pip<25' setuptools==69.5.1 wheel
RUN python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
