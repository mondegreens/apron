# Docker Image Decisions — Resolved

## Decision 1: Base image approach

**Question:** Build from `nvidia/cuda:cudnn-devel` (like model-serve) or
layer on `vllm/vllm-openai` (official image)?

**Answer:** Build from `nvidia/cuda:cudnn-devel`.

**Evidence:**

The official vLLM image uses `cuda-base` + a curated list of JIT
compilation packages (cuda-nvcc, cuda-nvrtc-dev, libcublas-dev,
libcurand-dev). This has broken multiple times in 2026:

- [#42291](https://github.com/vllm-project/vllm/issues/42291): v0.20.1/v0.20.2 — FlashInfer headers missing, "No such file or directory"
- [#42906](https://github.com/vllm-project/vllm/issues/42906): v0.19.1/v0.16.0 — pre-fix FlashInfer causes CUDA IMA crashes on MoE FP8
- [#44305](https://github.com/vllm-project/vllm/issues/44305): Blackwell — CUDA compiler/toolkit header incompatibility
- [#41865](https://github.com/vllm-project/vllm/issues/41865): FlashInfer JIT causes multi-worker deadlock
- [GHSA-jrf6-vqxq-pjv2](https://github.com/vllm-project/vllm/security/advisories/GHSA-jrf6-vqxq-pjv2): dependency confusion in flashinfer-jit-cache

The failure pattern: the curated list misses a header or library that
FlashInfer/DeepGEMM JIT compilation needs. Each new vLLM release can
introduce a new JIT dependency that wasn't in the curated list.

FlashInfer JIT requires ([sgl-project/sglang#5389](https://github.com/sgl-project/sglang/issues/5389), [flashinfer-ai/flashinfer#3493](https://github.com/flashinfer-ai/flashinfer/issues/3493)):
- nvcc (compiler)
- CUDA headers (cuda_fp8.h)
- CCCL (CUDA C++ Core Library / libcudacxx)
- CUTLASS headers
- nvrtc + nvrtc-dev

`cudnn-devel` includes ALL of these by definition. The curated list
in the official image includes SOME of them and has historically been
incomplete.

model-serve (vladryzhkov/vllm-model-serve) uses `cudnn-devel` and
has been proven on RunPod with real GPU deployments. Zero JIT failures.

**Cost:** ~5 GB larger image (cudnn-devel vs base). Paid once per host.
The JIT failure debugging + re-deployment cost is paid every time the
curated list is incomplete.

**Counterargument considered:** RunPod's own worker-vllm (runpod-workers/worker-vllm)
uses `FROM vllm/vllm-openai` as base and defaults to v0.29.0. But
RunPod's worker is for their serverless platform with pre-cached images
and their own support pipeline. Apron needs to work on RunPod, Modal,
on-prem — without a vendor support pipeline to catch JIT failures.

## Decision 2: Image source

**Question:** Where does the Dockerfile come from?

**Answer:** Fork model-serve's Dockerfile structure. Update version pins
to match vLLM v0.29.0's dependency chain. Strip model-serve-specific
deps (OCR, vision, chat templates). Keep everything that makes it work
on cloud GPUs (SSH, NCCL tuning, model download, env-var config,
cudnn-devel base).

**Reason:** model-serve's Dockerfile is battle-tested on RunPod. The
structure (two-stage build, venv copy, start.sh orchestration, SSH
setup, NCCL settings) transfers directly. The version re-pin to v0.29.0
is the work — the structure is proven.

## Decision 3: Registry

**Question:** Where is the image published?

**Answer:** DockerHub under `mondegreens/apron-runner:<vllm-version>`.
Also GHCR for non-RunPod providers. RunPod requires public DockerHub
images.

## Decision 4: Version tagging

**Question:** How is the image tagged?

**Answer:** `apron-runner:v0.29.0` — the tag IS the vLLM version. When
apron updates its constraint set, the tag changes. The image digest is
recorded in the execution fingerprint.

## Decision 5: Provider agnosticism

**Question:** Is the image RunPod-specific?

**Answer:** No. Standard NVIDIA Container Toolkit conventions
(NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES). Works on any
Docker + GPU host. No RunPod SDK inside the container. SSH via
PUBLIC_KEY env var (standard pattern). RunPod-specific code lives in
the RunPod adapter (src/apron/adapters/backends/runpod.py), not in
the image.

## Decision 6: start.sh entrypoint

**Question:** How does vLLM start?

**Answer:** `vllm serve` (v0.29.0, not deprecated api_server). Through
tini (PID 1 zombie reaper). Logs via tee to both stdout and
/var/log/vllm.log (SSH reads the file, no pipe racing). Full env-var
set supporting text, MoE, quantized, multimodal, LoRA models.

## Decision 7: What NOT to include

- nginx (RunPod proxy handles HTTPS)
- RunPod SDK (belongs in adapter, not image)
- Vision/OCR deps (add when needed, not preemptively)
- vim/htop/tmux/screen (debugging convenience, not production need)
