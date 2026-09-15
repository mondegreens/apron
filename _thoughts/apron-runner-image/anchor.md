---
problem: apron-runner-image
date: 2026-09-14
---

# Problem Statement

Apron needs its own Docker image for GPU evidence collection. The image
must run on any GPU cloud provider (RunPod today, others later), pin the
vLLM engine version to match apron's constraint set, handle model
download and vLLM boot via environment variables, provide SSH access for
profiling and management commands, and be publicly available on DockerHub
(RunPod cannot pull private images).

# What exists today

- `vladryzhkov/vllm-model-serve:latest` — a production image built for
  model-serve (docflux/model-serve). Pins vLLM 0.13.0. Handles SSH,
  model download, env-var-driven vLLM config. Works on RunPod. But:
  wrong vLLM version for apron's constraints, maintained for model-serve's
  purposes, not apron's.
- `vllm/vllm-openai:v0.29.0` — official vLLM image. Right version but no
  SSH, no model download, no start.sh orchestration. Not usable as-is.
- Apron's constraint set (calculator, log parsing, architecture dispatch)
  was built against vLLM v0.29.0 source files.

# Proposed solution

A Dockerfile in apron's repo that:
1. Pins vLLM to the version matching apron's constraint set
2. Includes SSH, model download, env-var config (inspired by model-serve)
3. Is pushed to DockerHub under apron's namespace
4. Is provider-agnostic (RunPod today, any provider with Docker support later)
5. Records the image tag + digest in the execution fingerprint

# Claimed benefits

- Evidence chain is honest: prediction version matches measurement version
- Apron controls its own infrastructure dependency
- Provider-agnostic: same image works on RunPod, Lambda, CoreWeave, etc.
- Reproducible: pinned versions, tagged builds, digest-addressable

# Load-bearing assumptions (attack targets)

1. vLLM v0.29.0 exists and can be installed via pip (not just from source)
2. The model-serve start.sh approach works with v0.29.0 (API compatibility)
3. DockerHub public images are accessible from all GPU cloud providers
4. SSH is the right management plane (vs HTTP-based alternatives)
5. A single image serves both verification (teardown) and deployment (keep-alive) modes
6. The image build can be automated in CI
7. vLLM's memory profiling log format at v0.29.0 matches what apron parses

# Acceptance criteria

1. Image builds from Dockerfile in apron's repo
2. Image is pushed to DockerHub and pullable by RunPod
3. `apron verify` with this image produces a VerificationReport with
   correct execution_fingerprint (engine version matches constraint set)
4. SSH access works for profiling commands
5. vLLM boots and serves health endpoint within 5 minutes of pod creation
6. Deterministic test cases pass (model responds correctly)
7. Image tag is recorded in execution fingerprint
8. Same image works on at least 2 different GPU types (e.g., L4 and A5000)
