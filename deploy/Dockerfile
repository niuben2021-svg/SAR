# Pass an official GPU PyTorch image by digest using scripts/offline/build.sh.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
LABEL org.opencontainers.image.title="SAR-VLM offline MVP" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
WORKDIR /opt/sar-vlm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/outputs/cache/huggingface MODELSCOPE_CACHE=/outputs/cache/modelscope \
    XDG_CACHE_HOME=/outputs/cache MPLCONFIGDIR=/outputs/cache/matplotlib \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled
# The context is an explicit allowlist in .dockerignore (no data, weights or credentials).
COPY . .
# A single preparation layer also works on Docker hosts using the space-heavy vfs driver.
RUN python -c "import sys,torch,torchvision; assert sys.version_info >= (3,11); assert torch.version.cuda; open('/tmp/sar-base-constraints.txt','w').write('torch=='+torch.__version__+'\\n'+'torchvision=='+torchvision.__version__+'\\n')" \
    && python -m pip install --no-cache-dir -c /tmp/sar-base-constraints.txt -r deploy/requirements-container.txt \
    && python -m pip install --no-cache-dir --no-deps . \
    && python -m pip check \
    && python -c "from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLVideoProcessor; from swift.arguments import SftArguments" \
    && python -m pytest -q -m 'not gpu' \
    && python -m pip freeze > /opt/sar-vlm/BUILD-ENVIRONMENT.txt \
    && cp /tmp/sar-base-constraints.txt /opt/sar-vlm/BASE-TORCH.txt \
    && chmod +x /opt/sar-vlm/deploy/entrypoint.sh
ENTRYPOINT ["/opt/sar-vlm/deploy/entrypoint.sh"]
CMD ["help"]
