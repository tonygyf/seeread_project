# SeeRead — low-light scene text spotting: GPU training / evaluation image
#
# Build:
#   docker build -t seeread:latest .
# Host requirements: NVIDIA driver + NVIDIA Container Toolkit (CUDA/cuDNN
# libraries are provided by the base image, no local CUDA install needed).
# torchvision.ops.deform_conv2d used by SnakeConvLite ships prebuilt in the
# base image, so the runtime (non-devel) variant is sufficient.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Optional PyPI mirror, e.g. https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=
# Non-root user UID/GID (override via compose to match the host user)
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface

# System deps: libglib2.0-0 for opencv-python-headless, git for baseline repos
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Non-root user so bind-mounted files under ./data and ./results keep host
# ownership on Linux; reuse the UID/GID if the base image already defines them.
RUN if getent group ${USER_GID} >/dev/null; then \
        APP_GROUP=$(getent group ${USER_GID} | cut -d: -f1); \
    else \
        groupadd -g ${USER_GID} appuser && APP_GROUP=appuser; \
    fi; \
    if id -u ${USER_UID} >/dev/null 2>&1; then \
        APP_USER=$(getent passwd ${USER_UID} | cut -d: -f1); \
    else \
        useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash appuser && APP_USER=appuser; \
    fi; \
    mkdir -p /home/${APP_USER}/.cache/huggingface /home/${APP_USER}/.cache/torch; \
    chown -R ${APP_USER}:${APP_GROUP} /home/${APP_USER}

WORKDIR /workspace

# Install Python deps first for layer caching. torch/torchvision in the base
# image already satisfy requirements.txt (>=2.1 / >=0.16), so they are not
# re-downloaded.
COPY --chown=appuser:appuser requirements.txt ./
RUN if [ -n "${PIP_INDEX_URL}" ]; then \
        pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# Project source. data/ and results/ are excluded via .dockerignore; recreate
# them here as empty mount points with correct ownership.
COPY --chown=appuser:appuser . /workspace/
RUN mkdir -p /workspace/data/raw \
             /workspace/data/annotations \
             /workspace/data/images \
             /workspace/data/images_lit \
             /workspace/results \
    && chown -R appuser:appuser /workspace

USER appuser

# Default follows the quick-start in docs/01_research_design.md.
# Override with any tools/ command, e.g.:
#   docker compose run --rm seeread python tools/legacy/train.py --config configs/legacy/seeread_joint.yaml
CMD ["python", "tests/test_smoke.py"]
