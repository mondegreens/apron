#!/bin/bash
set -e

# --------------------------------------------------------------------------- #
# SSH (key-only authentication via PUBLIC_KEY env var)
# --------------------------------------------------------------------------- #

if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh

    for type in rsa ecdsa ed25519; do
        [ ! -f /etc/ssh/ssh_host_${type}_key ] && \
            ssh-keygen -t $type -f /etc/ssh/ssh_host_${type}_key -q -N ''
    done

    service ssh start
fi

# --------------------------------------------------------------------------- #
# Export env vars for SSH sessions (container env vars are not inherited)
# --------------------------------------------------------------------------- #

printenv | grep -E '^VLLM_|^HF_|^NVIDIA_|^NCCL_|^CUDA_|^PATH=|^LD_LIBRARY_PATH=' \
    | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/apron_environment
echo 'source /etc/apron_environment' >> ~/.bashrc

# --------------------------------------------------------------------------- #
# NVIDIA / NCCL (tuned for cloud GPU providers)
# --------------------------------------------------------------------------- #

: "${NVIDIA_VISIBLE_DEVICES:=all}"
: "${NVIDIA_DRIVER_CAPABILITIES:=compute,utility}"
export NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES

: "${NCCL_IB_DISABLE:=1}"
: "${NCCL_SOCKET_IFNAME:=eth0}"
export NCCL_IB_DISABLE NCCL_SOCKET_IFNAME

[ "$ENABLE_MEMORY_FRAGMENTATION_FIX" = "1" ] && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --------------------------------------------------------------------------- #
# vLLM environment
# --------------------------------------------------------------------------- #

: "${VLLM_LOGGING_LEVEL:=INFO}"
export VLLM_LOGGING_LEVEL

[ -n "$VLLM_ATTENTION_BACKEND" ]            && export VLLM_ATTENTION_BACKEND
[ "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" = "1" ]  && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
[ -n "$VLLM_WORKER_MULTIPROC_METHOD" ]      && export VLLM_WORKER_MULTIPROC_METHOD

# --------------------------------------------------------------------------- #
# Model download (parallel downloader for pre-signed URLs)
# --------------------------------------------------------------------------- #

if [ "$DOWNLOAD_MODEL_FILES" = "true" ]; then
    echo "Starting parallel model file download"
    : "${DOWNLOAD_MAX_WORKERS:=32}"
    export DOWNLOAD_MAX_WORKERS
    python /workspace/parallel_downloader.py || exit 1
    echo "Model files downloaded"
fi

# --------------------------------------------------------------------------- #
# Build vllm serve arguments from env vars
# --------------------------------------------------------------------------- #

ARGS=""

# Core model
[ -n "$VLLM_MODEL" ]                       && ARGS="$ARGS $VLLM_MODEL"
[ -n "$VLLM_TOKENIZER" ]                   && ARGS="$ARGS --tokenizer $VLLM_TOKENIZER"
[ -n "$VLLM_DTYPE" ]                       && ARGS="$ARGS --dtype $VLLM_DTYPE"

# Quantization
[ -n "$VLLM_QUANTIZATION" ]                && ARGS="$ARGS --quantization $VLLM_QUANTIZATION"
[ -n "$VLLM_LOAD_FORMAT" ]                 && ARGS="$ARGS --load-format $VLLM_LOAD_FORMAT"

# Memory
[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]      && ARGS="$ARGS --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION"
[ -n "$VLLM_MAX_MODEL_LEN" ]               && ARGS="$ARGS --max-model-len $VLLM_MAX_MODEL_LEN"
[ -n "$VLLM_CPU_OFFLOAD_GB" ]              && ARGS="$ARGS --cpu-offload-gb $VLLM_CPU_OFFLOAD_GB"
[ -n "$VLLM_BLOCK_SIZE" ]                  && ARGS="$ARGS --block-size $VLLM_BLOCK_SIZE"
[ -n "$VLLM_KV_CACHE_DTYPE" ]             && ARGS="$ARGS --kv-cache-dtype $VLLM_KV_CACHE_DTYPE"

# Parallelism
[ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]        && ARGS="$ARGS --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE"
[ -n "$VLLM_PIPELINE_PARALLEL_SIZE" ]      && ARGS="$ARGS --pipeline-parallel-size $VLLM_PIPELINE_PARALLEL_SIZE"

# Scheduling
[ -n "$VLLM_MAX_NUM_SEQS" ]                && ARGS="$ARGS --max-num-seqs $VLLM_MAX_NUM_SEQS"
[ -n "$VLLM_MAX_NUM_BATCHED_TOKENS" ]      && ARGS="$ARGS --max-num-batched-tokens $VLLM_MAX_NUM_BATCHED_TOKENS"

# Model-specific
[ -n "$VLLM_REASONING_PARSER" ]            && ARGS="$ARGS --reasoning-parser $VLLM_REASONING_PARSER"
[ -n "$VLLM_CHAT_TEMPLATE" ]               && ARGS="$ARGS --chat-template $VLLM_CHAT_TEMPLATE"

# Boolean flags
[ "$VLLM_ENFORCE_EAGER" = "1" ]            && ARGS="$ARGS --enforce-eager"
[ "$VLLM_TRUST_REMOTE_CODE" = "1" ]        && ARGS="$ARGS --trust-remote-code"
[ "$VLLM_ENABLE_PREFIX_CACHING" = "1" ]    && ARGS="$ARGS --enable-prefix-caching"
[ "$VLLM_ENABLE_CHUNKED_PREFILL" = "1" ]   && ARGS="$ARGS --enable-chunked-prefill"
[ "$VLLM_MULTIMODAL" = "1" ]               && ARGS="$ARGS --multimodal"

# LoRA
[ "$VLLM_ENABLE_LORA" = "1" ]             && ARGS="$ARGS --enable-lora"
[ -n "$VLLM_MAX_LORAS" ]                   && ARGS="$ARGS --max-loras $VLLM_MAX_LORAS"
[ -n "$VLLM_MAX_LORA_RANK" ]              && ARGS="$ARGS --max-lora-rank $VLLM_MAX_LORA_RANK"

# S3 model loading
if [ "$VLLM_LOAD_FORMAT" = "runai_streamer" ]; then
    [ -n "$AWS_ENDPOINT_URL" ] && echo "Using S3 endpoint: $AWS_ENDPOINT_URL"
fi

# --------------------------------------------------------------------------- #
# HuggingFace token
# --------------------------------------------------------------------------- #

if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN
fi

# --------------------------------------------------------------------------- #
# Launch vLLM
# --------------------------------------------------------------------------- #

SAFE_ARGS=$(echo "$ARGS" | sed 's/--hf-token [^ ]*/--hf-token ***REDACTED***/g')
echo "Starting vLLM: vllm serve $SAFE_ARGS"

set -o pipefail
exec vllm serve $ARGS 2>&1 | tee /var/log/vllm.log
