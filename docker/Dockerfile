# syntax=docker/dockerfile:1.7
# APRON_RUNNER_VERSION=v1.0.0
# vLLM engine version pinned to match apron's constraint set.
#
# Architecture: two-stage build from nvidia/cuda cudnn-devel base.
# cudnn-devel (not cuda-base) because FlashInfer and DeepGEMM JIT-compile
# CUDA kernels at serve time and need nvcc, CCCL headers, CUTLASS, and
# the full toolkit — the official vLLM image's curated minimal list has
# broken repeatedly (vllm-project/vllm#42291, #42906, #44305, #41865).
#
# Text, MoE, quantized, multimodal, and LoRA models via env vars.

ARG CUDA_VERSION=13.0.0
ARG UBUNTU_VERSION=22.04

###############################################################################
# STAGE 1 — builder
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential cmake ninja-build \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && --mount=type=cache,target=/tmp/uv-cache \
       uv pip install --cache-dir /tmp/uv-cache -r requirements.txt

COPY start.sh parallel_downloader.py ./
RUN chmod +x start.sh

###############################################################################
# STAGE 2 — runtime (cudnn-devel, not runtime — JIT compilation needs it)
###############################################################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        curl ca-certificates git \
        openssh-server tini \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd \
    && rm -f /etc/ssh/ssh_host_* \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/start.sh /start.sh
COPY --from=builder /workspace/parallel_downloader.py /workspace/parallel_downloader.py
RUN chmod +x /start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

VOLUME ["/root/.cache/huggingface"]
EXPOSE 22 8000

ENTRYPOINT ["tini", "-s", "--", "/start.sh"]
