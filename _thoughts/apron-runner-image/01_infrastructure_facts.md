# Infrastructure Facts (verified)

## vLLM v0.29.0
- Released: 2026-09-09 (5 days ago)
- PyPI: `pip install vllm==0.29.0`
- Docker: `vllm/vllm-openai:v0.29.0` on DockerHub
- Requires: CUDA 13.0
- Breaking: `python -m vllm.entrypoints.openai.api_server` deprecated → use `vllm serve`
- Breaking: 10 model architectures removed (need to verify Qwen3ForCausalLM survives)
- Source: https://freedom.tech/posts/2026-09-09-vllm-0-29-0/

## CUDA 13.0 Base Image
- Available: `nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04`
- BUT: vllm/vllm-openai:v0.29.0 already includes CUDA — no need for raw CUDA base

## Model-serve Image
- `vladryzhkov/vllm-model-serve:latest`
- vLLM 0.13.0 on CUDA 12.8
- Uses deprecated `python -m vllm.entrypoints.openai.api_server`
- Built from nvidia/cuda:12.8.0 base (full from-scratch build)
- SSH, nginx, parallel downloader, env-var config, HF transfer

## RunPod Constraints
- Can ONLY pull public DockerHub images
- Passes env vars at pod creation
- SSH via PUBLIC_KEY env var
- Proxy URL: https://{pod_id}-{port}.proxy.runpod.net

## Architecture Decision
Two viable approaches:

**Option A — Layer on official vLLM image:**
```
FROM vllm/vllm-openai:v0.29.0
+ SSH server
+ start.sh (from model-serve, adapted)
+ env-var-driven config
```
Pro: vLLM team handles CUDA/torch/engine complexity
Pro: Small image delta (just SSH + orchestration)
Con: Depends on vLLM's image release pipeline (issue #33748: sometimes broken)

**Option B — Build from CUDA base (like model-serve):**
```
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04
+ Python, vLLM==0.29.0, torch, all deps
+ SSH, start.sh, etc.
```
Pro: Full control over every dependency
Con: Much larger Dockerfile, complex dependency chain, longer build time
Con: Rebuilds what vLLM already does

Option A is the responsible choice unless the vLLM image is broken.
