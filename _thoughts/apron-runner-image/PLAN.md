# Apron Runner Image — Implementation Plan

**Verdict:** Build with revisions. Fork model-serve's proven Dockerfile
structure (battle-tested on RunPod: SSH, NCCL tuning, model download,
env-var config, multi-stage build). Update version pins to vLLM v0.29.0
and its dependency chain. Strip model-serve-specific deps (OCR, vision,
custom chat templates). Published to DockerHub (RunPod requires public
images) and GHCR (for non-RunPod providers). Version-tagged to match
the pinned vLLM version.

**Why fork model-serve, not layer on official image:** The official
`vllm/vllm-openai:v0.29.0` has never been tested on RunPod with SSH,
NCCL cloud tuning, or model-download orchestration. model-serve's
Dockerfile has been. NCCL settings, TORCH_CUDA_ARCH_LIST, memory
fragmentation fix, SSH setup — every line earned its place through
real deployment failures. The version re-pin is the hard part; the
proven infrastructure transfers directly.

**Effort:** ~6 hours implementation (version re-pin is the majority),
~$0.50 GPU validation.

---

## Layer 0 — Measure + Prove (~1 hour)

### 0.1 Extract vLLM v0.29.0's actual dependency requirements

```bash
# From the vLLM source at the pinned tag
cd .sources/vllm
git checkout v0.29.0
cat pyproject.toml | grep -A 50 "dependencies"
cat requirements/build.txt
cat requirements/common.txt
```

Determine: exact torch version, CUDA version, flashinfer version,
transformers version, and all other pinned deps that vLLM v0.29.0
needs. This is the source of truth — not guessing or hand-editing.

**GO:** Dependencies extracted, CUDA base version identified.
**NO-GO:** v0.29.0 tag doesn't exist in .sources/vllm → fetch it.

### 0.2 Verify CUDA base image exists

From 0.1, determine the required CUDA version (likely 13.0.x).
```bash
docker pull nvidia/cuda:<version>-cudnn-devel-ubuntu22.04
```

**GO:** Image pulls successfully.

### 0.3 Verify Qwen3ForCausalLM is supported in v0.29.0

```bash
# Check if the architecture survived v0.29.0's breaking changes
grep -r "Qwen3ForCausalLM\|Qwen3" .sources/vllm/vllm/model_executor/models/
```

**GO:** Architecture present.
**NO-GO:** Removed in v0.29.0 → check what replaced it, update fixtures.

### 0.4 Verify Docker build environment

```bash
# Check Mac Mini
ssh vladryzhkov@100.94.88.37 "docker buildx version && docker info | grep -i platform"
# OR identify another build machine
```

**GO:** Docker + buildx available for linux/amd64.

---

## Layer 1 — Dockerfile (~2 hours)

File: `docker/Dockerfile`

Forked from model-serve's proven structure, updated for vLLM v0.29.0,
stripped of model-serve-specific deps.

```dockerfile
# syntax=docker/dockerfile:1.7
# APRON_RUNNER_VERSION=v1.0.0-vllm-0.29.0
###############################################################################
# STAGE 1 — builder
###############################################################################
FROM nvidia/cuda:<CUDA_VERSION>-cudnn-devel-ubuntu22.04 AS builder
# ^^^ CUDA version from Layer 0.1 — must match vLLM v0.29.0's requirement

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# uv for fast dependency resolution (from model-serve)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Python + venv
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Dependencies — pinned from vLLM v0.29.0's actual release requirements
COPY requirements.txt .
RUN python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && uv pip install --cache-dir /tmp/uv-cache -r requirements.txt

COPY start.sh parallel_downloader.py ./
RUN chmod +x start.sh

###############################################################################
# STAGE 2 — runtime (cuda-runtime, not devel — smaller image)
###############################################################################
FROM nvidia/cuda:<CUDA_VERSION>-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

WORKDIR /workspace

# Minimal runtime packages (from model-serve, stripped)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        curl ca-certificates git \
        openssh-server tini \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && rm -f /etc/ssh/ssh_host_* \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy venv + scripts from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/start.sh /start.sh
COPY --from=builder /workspace/parallel_downloader.py /workspace/parallel_downloader.py
RUN chmod +x /start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

VOLUME ["/root/.cache/huggingface"]
EXPOSE 22 8000

ENTRYPOINT ["tini", "-s", "--", "/start.sh"]
```

Key differences from model-serve's Dockerfile:
- **Runtime stage uses cuda-runtime, not cuda-devel** — model-serve
  keeps the full CUDA toolkit in runtime (the attacker flagged this).
  If vLLM v0.29.0 JIT-compiles at startup, fall back to devel.
- **No nginx, vim, htop, tmux, screen, rsync, sudo** — stripped
- **No vision/OCR deps** (opencv, mistral-common, easydict, addict)
- **No chat templates** — apron uses completion API, not chat
- **tini as PID 1** — model-serve doesn't have this (zombie leak risk)
- **uv cache mount** — model-serve only caches apt, not pip/uv
- **Logs via tee** — start.sh writes to /var/log/vllm.log (no pipe race)

What's KEPT from model-serve (proven on RunPod):
- Two-stage builder→runtime venv-copy pattern
- TORCH_CUDA_ARCH_LIST covering 8.0/8.6/8.9/9.0 (all RunPod GPUs)
- SSH with PUBLIC_KEY env var injection
- NCCL settings (NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME)
- HF_HUB_ENABLE_HF_TRANSFER for fast model download
- Env-var-to-CLI-arg translation in start.sh
- parallel_downloader.py (portable, vLLM-agnostic)
- HEALTHCHECK for orchestrator readiness

### requirements.txt

**NOT hand-edited.** Generated from vLLM v0.29.0's actual release:

```bash
# Extract from vLLM's own dependency spec
cd .sources/vllm && git checkout v0.29.0
# Combine pyproject.toml dependencies + requirements/*.txt
# Pin every transitive dep with: pip compile or uv pip compile
```

The exact contents depend on Layer 0.1's findings. The principle:
every version comes from vLLM's own release, not from guessing.

---

## Layer 2 — start.sh (~1 hour)

File: `docker/start.sh`

Adapted from model-serve's start.sh, stripped to essentials:

```bash
#!/bin/bash
set -e

# --- SSH setup (key-only, from PUBLIC_KEY env var) ---
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

# --- NVIDIA environment ---
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}

# --- NCCL settings (proven on cloud GPU providers) ---
: "${NCCL_IB_DISABLE:=1}"
: "${NCCL_SOCKET_IFNAME:=eth0}"
export NCCL_IB_DISABLE NCCL_SOCKET_IFNAME

# --- Memory fragmentation fix ---
[ "$ENABLE_MEMORY_FRAGMENTATION_FIX" = "1" ] && \
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- vLLM environment ---
: "${VLLM_LOGGING_LEVEL:=INFO}"
export VLLM_LOGGING_LEVEL
[ -n "$VLLM_ATTENTION_BACKEND" ] && export VLLM_ATTENTION_BACKEND
[ "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" = "1" ] && export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# --- Model download (if parallel downloader configured) ---
if [ "$DOWNLOAD_MODEL_FILES" = "true" ]; then
    python /workspace/parallel_downloader.py || exit 1
fi

# --- Build vllm serve arguments from env vars ---
# Full set — supports text, MoE, quantized, multimodal, LoRA models
ARGS=""
[ -n "$VLLM_MODEL" ]                     && ARGS="$ARGS $VLLM_MODEL"
[ -n "$VLLM_TOKENIZER" ]                && ARGS="$ARGS --tokenizer $VLLM_TOKENIZER"
[ -n "$VLLM_DTYPE" ]                     && ARGS="$ARGS --dtype $VLLM_DTYPE"
[ -n "$VLLM_QUANTIZATION" ]             && ARGS="$ARGS --quantization $VLLM_QUANTIZATION"
[ -n "$VLLM_LOAD_FORMAT" ]              && ARGS="$ARGS --load-format $VLLM_LOAD_FORMAT"
[ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]    && ARGS="$ARGS --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION"
[ -n "$VLLM_MAX_MODEL_LEN" ]            && ARGS="$ARGS --max-model-len $VLLM_MAX_MODEL_LEN"
[ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]     && ARGS="$ARGS --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE"
[ -n "$VLLM_PIPELINE_PARALLEL_SIZE" ]   && ARGS="$ARGS --pipeline-parallel-size $VLLM_PIPELINE_PARALLEL_SIZE"
[ -n "$VLLM_MAX_NUM_SEQS" ]             && ARGS="$ARGS --max-num-seqs $VLLM_MAX_NUM_SEQS"
[ -n "$VLLM_MAX_NUM_BATCHED_TOKENS" ]   && ARGS="$ARGS --max-num-batched-tokens $VLLM_MAX_NUM_BATCHED_TOKENS"
[ -n "$VLLM_BLOCK_SIZE" ]               && ARGS="$ARGS --block-size $VLLM_BLOCK_SIZE"
[ -n "$VLLM_CPU_OFFLOAD_GB" ]           && ARGS="$ARGS --cpu-offload-gb $VLLM_CPU_OFFLOAD_GB"
[ -n "$VLLM_REASONING_PARSER" ]         && ARGS="$ARGS --reasoning-parser $VLLM_REASONING_PARSER"
[ "$VLLM_ENFORCE_EAGER" = "1" ]         && ARGS="$ARGS --enforce-eager"
[ "$VLLM_TRUST_REMOTE_CODE" = "1" ]     && ARGS="$ARGS --trust-remote-code"
[ "$VLLM_ENABLE_PREFIX_CACHING" = "1" ] && ARGS="$ARGS --enable-prefix-caching"
[ "$VLLM_ENABLE_CHUNKED_PREFILL" = "1" ] && ARGS="$ARGS --enable-chunked-prefill"
[ "$VLLM_MULTIMODAL" = "1" ]            && ARGS="$ARGS --multimodal"
[ "$VLLM_ENABLE_LORA" = "1" ]          && ARGS="$ARGS --enable-lora"
[ -n "$VLLM_MAX_LORAS" ]               && ARGS="$ARGS --max-loras $VLLM_MAX_LORAS"
[ -n "$VLLM_MAX_LORA_RANK" ]           && ARGS="$ARGS --max-lora-rank $VLLM_MAX_LORA_RANK"

# --- HuggingFace token ---
[ -n "$HF_TOKEN" ] && export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" && export HF_TOKEN

# --- Launch vLLM (v0.29.0 uses `vllm serve`, not deprecated api_server) ---
# tee writes to both stdout (container logs / RunPod viewer) and a file
# (readable via SSH without racing the pipe). set -o pipefail ensures
# vllm's exit code propagates through the pipeline.
set -o pipefail
echo "Starting vLLM: vllm serve $ARGS"
exec vllm serve $ARGS 2>&1 | tee /var/log/vllm.log
```

Key differences from model-serve's start.sh:
- Uses `vllm serve` (v0.29.0) instead of deprecated `python -m vllm.entrypoints.openai.api_server`
- `tee /var/log/vllm.log` — logs go to both stdout (RunPod log viewer)
  AND a file (SSH reads `/var/log/vllm.log`, no pipe racing with
  `/proc/1/fd/1`)
- `tini` as PID 1 (in ENTRYPOINT) reaps zombies from orphaned SSH
  sessions. vLLM runs as tini's child, not as PID 1 itself.
- No nginx (unnecessary proxy layer)
- No parallel_downloader (vLLM + hf-transfer handles download)
- No RunPod-specific env export to `/etc/rp_environment`
- Password auth disabled — key-only SSH

---

## Layer 3 — Build + Push (~30 min)

### 3.1 DockerHub

Registry: `mondegreens/apron-runner` (matches the GitHub org)

```bash
docker build -t mondegreens/apron-runner:v0.29.0 \
  --build-arg VLLM_VERSION=v0.29.0 \
  -f docker/Dockerfile docker/

docker push mondegreens/apron-runner:v0.29.0
```

Tag convention: `v{vllm_version}` — the runner tag IS the engine
version. When apron updates constraints, the tag changes.

### 3.2 GHCR (for non-RunPod providers)

```bash
docker tag mondegreens/apron-runner:v0.29.0 \
  ghcr.io/mondegreens/apron-runner:v0.29.0

docker push ghcr.io/mondegreens/apron-runner:v0.29.0
```

### 3.3 Update apron code

Update `DEFAULT_IMAGE` in `src/apron/adapters/backends/runpod.py`:
```python
DEFAULT_IMAGE = "mondegreens/apron-runner:v0.29.0"
```

Update `RunPodTarget.build_env()` to use `vllm serve` (already
correct since the start.sh handles the entrypoint).

Update `VllmEngineAdapter.verify()` log reading:
```python
# OLD (pipe racing bug — /proc/1/fd/1 is a pipe shared with container runtime):
log_result = target.execute("cat /proc/1/fd/1 2>/dev/null || ...")

# NEW (reads the tee'd log file — no pipe racing):
log_result = target.execute("cat /var/log/vllm.log")
```

### 3.4 CI automation (Phase 2)

A GitHub Action that:
1. Builds the image on tag push
2. Pushes to DockerHub + GHCR
3. Runs Layer 0 validation on a GPU runner

Not required for Phase 1a — manual build is acceptable.

---

## Layer 4 — Provider Abstraction (~1 hour)

The image must work beyond RunPod. The design (ADR-001, framework-spec)
requires at least two backends.

### 4.1 Image works on any Docker + NVIDIA GPU host

The image uses standard NVIDIA Container Toolkit conventions:
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- No RunPod-specific APIs or paths inside the container

A local Docker run:
```bash
docker run --gpus all -p 8000:8000 -p 2222:22 \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_rsa.pub)" \
  -e VLLM_MODEL=Qwen/Qwen3-8B \
  -e VLLM_DTYPE=bfloat16 \
  -e VLLM_MAX_MODEL_LEN=640 \
  mondegreens/apron-runner:v0.29.0
```

Works on local workstation, bare metal, Lambda, CoreWeave — anywhere
with Docker + NVIDIA GPU runtime.

### 4.2 RunPod adapter passes the image + env vars

Already implemented in `RunPodTarget.provision()` — passes `image_name`
and `env` dict to `runpod.create_pod()`.

### 4.3 Future: Modal adapter

Modal uses a different container model (Functions, not persistent pods).
The same image can be used with `modal.Image.from_registry()`.
Not Phase 1a scope, but the image design doesn't block it.

---

## Layer 5 — Validation on Real GPU (~$0.50, ~30 min)

### 5.1 Build and push the image

**Pre-check:** Verify Docker is available on the build machine.

```bash
# Option A: Mac Mini (if Docker + buildx are installed)
ssh vladryzhkov@100.94.88.37 "docker buildx version" 2>&1

# Option B: GitHub Actions (no local Docker needed)
# Push docker/ files to the branch and trigger a manual workflow

# Option C: Any Linux machine with Docker
docker build -t mondegreens/apron-runner:v0.29.0 \
  --build-arg VLLM_VERSION=v0.29.0 \
  -f docker/Dockerfile docker/
docker push mondegreens/apron-runner:v0.29.0
```

The build is linux/amd64 only. A Mac can cross-build with
`docker buildx build --platform linux/amd64` but cannot test GPU
functionality locally — GPU validation happens in Layer 5.2 on RunPod.

**Image size:** The base `vllm/vllm-openai:v0.29.0` is ~8 GB
compressed. Apron's layer adds ~30 MB (tini + openssh + hf-transfer).
Total: ~8 GB pull. This is a cold-start cost per RunPod host —
subsequent pods on the same host reuse the cached image. For Phase 1a
(infrequent runs), this is acceptable. Phase 2 optimization: use a
RunPod template with pre-warmed image cache.

### 5.2 Run on RunPod

```python
pod = runpod.create_pod(
    name="apron-runner-test",
    image_name="mondegreens/apron-runner:v0.29.0",
    gpu_type_id=selected_gpu,
    ports="22/tcp,8000/http",
    env={
        "VLLM_MODEL": "Qwen/Qwen3-8B",
        "VLLM_DTYPE": "bfloat16",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
        "VLLM_MAX_MODEL_LEN": "640",
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "PUBLIC_KEY": public_key,
    },
)
```

### 5.3 Verify

1. `/health` returns 200 (vLLM booted)
2. `/version` returns `{"version": "0.29.0"}` (correct engine)
3. `/v1/completions` returns correct output for test case
4. SSH connects with key auth
5. `python3 -c "import torch; print(torch.cuda.mem_get_info())"` works via SSH
6. vLLM startup logs contain memory profiling data parseable by
   `VllmEngineAdapter.parse_profiling_logs()`
7. Teardown terminates the pod

### 5.4 GO/NO-GO

**GO:** All 7 checks pass. Proceed with fixture run using this image.

**NO-GO on check 1-3:** vLLM v0.29.0 has a boot issue on this GPU →
investigate, may need `--enforce-eager` or different `max_model_len`.

**NO-GO on check 6:** Log format changed from what .sources/vllm shows →
update `parse_profiling_logs()` regexes against real v0.29.0 output.

---

## What NOT to build

- **No parallel_downloader.py** — hf-transfer + vLLM's native download
  is fast enough for Phase 1a. Optimize later if download time > 5 min.
- **No nginx** — RunPod proxy handles external HTTPS termination.
- **No vision/OCR deps** — text generation only in Phase 1a.
- **No CI image build** — manual build is acceptable for Phase 1a.
  Automate when the image changes frequently (Phase 2 vLLM version bumps).
- **No multi-arch** — amd64 only. RunPod and all current GPU providers
  are x86_64.

---

## Files to create

| File | Purpose |
|---|---|
| `docker/Dockerfile` | Image definition, layers on vllm/vllm-openai |
| `docker/start.sh` | SSH setup + env-to-args + exec vllm serve |
| Update `src/apron/adapters/backends/runpod.py` | DEFAULT_IMAGE → mondegreens/apron-runner:v0.29.0 |

---

## Verification gate

Phase 1a exit gate is met when:
1. `apron verify` uses the apron-runner image (not model-serve or raw vllm)
2. `/version` in the VerificationReport matches v0.29.0
3. The execution fingerprint includes the correct engine version
4. The prediction-vs-measured comparison uses matching constraint sets
5. The image is publicly pullable from DockerHub
