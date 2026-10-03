# KV budget: predicted minus measured, term by term

2026-09-28.  GPU-free: stored verification reports, the pinned vLLM sources
(`.sources/vllm` = v0.29.0, `.sources/vllm-v0.30.0`), the peak probes in
`peak-probe/`.  Regenerate the "after" table with
`uv run python scripts/kv_budget_residuals.py`.

## What vLLM subtracts

`determine_available_memory`, identical in v0.29.0 (`v1/worker/gpu_worker.py:513-610`)
and v0.30.0 (`:528-625`):

    available = requested - non_kv_cache_memory - cudagraph_estimate_applied
    requested = ceil(total x gpu_memory_utilization)          v1/worker/utils.py:521-523 / 539-541
    non_kv_cache_memory = total_consumed + transient_peak_headroom   utils/mem_utils.py:317-326

- `total_consumed` is free memory lost since the snapshot vLLM takes *after* the CUDA
  context and NCCL are initialised (`gpu_worker.py:414-440` / `432-451`): neither is
  charged to the request.
- The CUDA-graph term is vLLM's *estimate* (`profile_cudagraph_memory`), not the memory
  the real capture later takes, and it is subtracted in both versions:
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` defaults to 1 (`envs.py:2105` / `2109`).
  The estimate is 1.0-2.6x the logged "actual".
- In the terms the report stores: `available = requested - weights - torch peak
  increase - non_torch - cuda_graph_estimate`, where Apron's `non_pytorch_increase` is
  `non_kv - weights - torch_peak_increase` (vllm_engine.py `_build_verification_report`).
  All 28 healthy memory records satisfy this to the log's 0.01 GiB
  (`test_records_satisfy_vllms_kv_budget_identity`).

## GPU totals: the exact bytes the pods detected

The records carry the total only as the log's rounded GiB.  The detected byte count is
recoverable: every report's `detected_hardware_fingerprint` is the multihash of
`HardwareSpec(gpu_sku, total_memory_bytes=props.total_memory, compute_capability)`.
Searching byte by byte within +-0.0051 GiB of the logged total, one value matches each:

| SKU | nominal (GPU_SPECS before) | detected | log |
|---|---|---|---|
| H100 80GB HBM3 | 85,899,345,920 (80 GiB) | 85,017,493,504 (81,079 MiB) | 79.18 |
| A100 80GB PCIe | 85,899,345,920 | 85,093,777,408 / 85,094,825,984 (two pods, 1 MiB apart) | 79.25 |
| A100-SXM4-80GB | 85,899,345,920 | 85,093,777,408 | 79.25 |
| RTX A6000 | 51,539,607,552 | 50,899,648,512 | 47.40 |
| RTX 4090 | 25,769,803,776 | 25,250,627,584 | 23.52 |
| L4 | 25,769,803,776 | 23,659,151,360 | 22.03 |

GPU_SPECS now holds the detected counts (the smaller A100 PCIe one: the request is never
over-stated).  Never detected, left nominal or as reported and marked so: RTX A5000, RTX
3090, L40, H200, B200.  `test_gpu_specs_are_the_byte_counts_the_pods_detected` re-hashes
them against the records.  No batch-tier threshold moves (H100 stays >= 70 GiB).

## Before: the stored deltas and today's calculator before these fixes

`stored` = the record's `predicted_minus_measured.kv_cache_consistency` (the claim made
at run time).  `before` = the calculator as it was this morning on the plan as booted (its
TP, dtype, utilization), split into the terms (positive: more KV predicted than vLLM
left).  GiB.

| model | GPU | TP | stored | before | total | weights | peak | non-torch | graphs |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-FP8 | H100 | 1 | +0.07 | +1.29 | +0.76 | +0.04 | +0.02 | +0.67 | -0.21 |
| Qwen3-1.7B (x2) | RTX 4090 | 1 | +0.11 | +0.47 | +0.44 | +0.02 | +0.00 | +0.41 | -0.40 |
| mamba-2.8b (float32 plan) | RTX 4090 | 1 | +5.75 | +0.34 | +0.44 | -0.00 | +0.00 | +0.19 | -0.29 |
| Qwen3-1.7B | L4 | 1 | +1.40 | +1.76 | +1.77 | +0.02 | +0.00 | +0.37 | -0.41 |
| Qwen3-32B | A100 PCIe | 1 | -4.10 | +0.52 | +0.67 | +0.01 | +0.00 | -0.20 | +0.02 |
| Qwen3-14B | A6000 | 1 | -2.13 | +0.09 | +0.55 | +0.01 | +0.00 | -0.23 | -0.25 |
| Qwen3-8B | RTX 4090 | 1 | -0.87 | -0.06 | +0.44 | +0.01 | -0.00 | -0.21 | -0.32 |
| Llama-3.1-8B | RTX 4090 | 1 | -0.55 | -0.07 | +0.44 | +0.04 | +0.00 | -0.23 | -0.32 |
| mamba-2.8b (bf16) | RTX 4090 | 1 | +0.13 | +0.13 | +0.44 | +0.07 | +0.00 | -0.07 | -0.32 |
| DeepSeek-V2-Lite (+Chat) | A100 PCIe | 1 | +0.80 | +0.67 | +0.67 | +0.06 | +0.00 | +0.09 | -0.17 |
| gemma-2-2b (x2) | RTX 4090 | 1 | +0.22 / +0.69 | +0.07 | +0.44 | +0.07 | +0.00 | -0.06 | -0.40 |
| Mistral-7B (x2) | RTX 4090 | 1 | -1.27 / -1.74 | -0.20 | +0.44 | +0.01 | -0.00 | -0.33 | -0.32 |
| Qwen3.6-35B-A3B-FP8 | H100 | 1 | +2.47 | +3.62 | +0.74 | +0.14 | +1.26 | +0.96 | +0.50 |
| GLM-4.7-Flash | H100 | 1 | +3.45 | +4.45 | +0.74 | +0.10 | +0.01 | +1.10 | +2.50 |
| Qwen3-8B | A100 SXM | 2 | -10.41 | +0.14 | +0.69 | +0.01 | +0.01 | -0.27 | -0.30 |
| Muse-Glimmer-30B | H100 | 1 | +1.63 | +3.05 | +0.74 | +0.37 | +0.55 | +0.82 | +0.58 |
| Mistral-7B | L4 | 1 | +0.03 | +1.10 | +1.77 | +0.01 | -0.00 | -0.36 | -0.32 |
| Qwen3-8B-GPTQ-Int4 | RTX 4090 | 1 | +0.59 | -0.04 | +0.44 | -0.00 | -0.00 | -0.18 | -0.29 |
| Qwen3-0.6B | RTX 4090 | 1 | +0.16 | +0.16 | +0.44 | +0.01 | +0.00 | +0.13 | -0.43 |
| gpt-oss-120b | H100 | 1 | +3.15 | +3.15 | +0.74 | +0.66 | +0.00 | +1.69 | +0.05 |
| Qwen3-8B-AWQ | H100 | 1 | -0.03 | +1.57 | +0.76 | +0.14 | +0.01 | +0.45 | +0.21 |
| Qwen3-0.6B-FP8 | RTX 4090 | 1 | -0.12 | -0.12 | +0.44 | +0.04 | +0.00 | -0.20 | -0.40 |
| gemma-4-31B-it | H100 | 1 | +0.97 | +2.28 | +0.74 | +0.53 | +0.01 | +0.97 | +0.01 |
| Qwen3-32B | H100 | 1 | -0.49 | +4.01 | +0.74 | +0.01 | +0.00 | +1.81 | +1.44 |

(The Qwen3.6 / Muse "peak" column is the encoder moment, fixed separately the same day
from the v0.30 probe; it is in "after".)

### The stored outliers are claims for another plan, not calculator errors

- **Qwen3-8B, A100-SXM, TP 2, -10.41.**  The stored claim was made for TP 1 (15.26 GiB of
  weights, no `tensor_parallel`) by an older planner and at utilization 0.90; the fix
  proof booted TP 2 at vLLM's default 0.92 (7.64 GiB per rank).  Today's calculator on the
  booted plan: -0.05.  TP was already passed to fix proofs; the utilization was not.
- **Qwen3-32B, A100, -4.10.**  The stored claim's activation was the old rule (6.10 GiB
  against 1.49 measured).  Today: -0.01.
- **Qwen3-14B, A6000, -2.13.**  Old activation rule (2.75 vs 1.49) and 0.90 predicted for a
  plan that booted at 0.92 (0.95 GiB).  Today: -0.01.
- **mamba-2.8b, float32 plan, +5.75.**  The claim was for a 16-bit load (5.16 GiB) while
  the fix-proof plan served float32 (10.31 GiB): `plan_for` predicted every given plan at
  the checkpoint's dtype=auto dtype and at 0.90.  Planner plans already name the served
  dtype (`served_dtype`).  Today: +0.15.
- **Fixed now:** `CohortPlanner.plan_for` predicts a given plan at its own dtype, its own
  utilization (vLLM's default 0.92, `config/cache.py:111` / `:103`, when it sets none) and
  its own `max_num_seqs` (`test_a_given_plan_is_predicted_at_its_dtype_and_utilization`).

## The causes, and what changed

### 1. Total memory (all GPUs): +0.44 to +1.77 GiB

GPU_SPECS nominal sizes over-stated `requested` by utilization x (nominal - detected).
Fixed with the detected counts above; `requested_memory_bytes` is vLLM's
`ceil(total x util)`.  Residual after: 0.00-0.01 GiB (the log's rounding).

### 2. Non-torch (the 500 MB constant): -0.36 to +1.81 GiB

What is in it, measured on the probes (vLLM v0.29 logs exact bytes there):

| probe | weights (logged) | allocated before the profile run | of which sampler state | non-torch logged |
|---|---|---|---|---|
| Gemma 4 31B | 63,340,030,197 | +1,247,013,131 | 1,140,850,688 | 1,546,188,228 |
| Gemma 4 31B language-model-only | 62,180,389,027 | +1,153,981,277 | 1,140,850,688 | 1,717,986,920 |
| GLM-4.7-Flash | 59,989,955,706 | +696,961,414 | 674,037,760 | 1,685,774,665 |

- **Sampler state** (model runner V2, the default in both versions,
  `config/vllm.py:645-677`): `output_bin_counts` [max_num_seqs, V] int32
  (`v1/worker/gpu/sample/penalties.py:40-42`), `prompt_bin_mask` and the structured-output
  `grammar_bitmask`, each [max_num_seqs, ceil(V/32)] int32 (`penalties.py:32-37`,
  `structured_outputs.py:51-53` / `50-52`).  Allocated after the weights profiler
  (`model_runner.py:380-403`, sampler at `:437`), so it lands in "non-torch".  Driver: vocab
  x max_num_seqs, i.e. the GPU's batch tier (1024 slots on an H100, 256 below 70 GiB).
  0.03 GiB for Mistral on a 4090, 1.06 GiB for Gemma 4 on an H100.  Exact on both probes.
- **Encoder embeddings buffer** for a vision/audio wrapper: [max_num_batched_tokens, H] in
  the model dtype (`v1/worker/gpu/mm/encoder_runner.py:58-60`): 88 MB of Gemma 4's
  106 MB remainder.
- **Base** (CUDA/library memory after the snapshot, allocator slack), what is left per
  record: 0.05-0.16 GiB on the sm8x boots (median 0.109), 0.21-0.39 GiB on the H100 boots
  (median 0.300).  Not NCCL, not the CUDA context (before the snapshot): TP 2 shows the
  smallest base, 0.05.  The two classes differ also in batch tier (all H100 boots are
  8192 tokens / 1024 seqs, all others 2048 / 256), so the records cannot say whether the
  step is Hopper or the tier; the calculator keys it on compute capability and takes the
  H100 value for an unknown GPU.
- **A held compile segment.**  In 10 of the 21 distinct TP 1 text-only boots, the non-torch
  residue is one more V x H x b block (0.90-1.17 of it) -- the size of the random tensor
  TorchInductor's combo-kernel benchmark allocates for the embedding (moment 1 of
  `_activation_estimate`) -- while no live torch tensor holds it after the run (the
  persistent growth during the run is 0.01-0.04 GiB outside MoE workspaces).  In the other
  10 it is absent, and the same checkpoint goes both ways on two GPUs (Qwen3-32B: A100 none,
  H100 1.45 GiB).  Cause as far as the records go: the caching allocator keeps a segment of
  that size reserved; which boots keep it is placement, not model.  Never in a vision
  wrapper (the embedding is not compiled) or at TP 2.  Treated as a per-plan uncertainty
  (`compile_segment_bytes`), not in the point estimate; the guard subtracts it.

  | held (share of V x H x b) | none |
  |---|---|
  | Qwen3-1.7B 4090 1.07, L4 1.00; Qwen3-0.6B 1.17; mamba bf16 1.00, fp32 1.04; DeepSeek-V2-Lite (+Chat) 0.90; GLM-4.7-Flash 1.08; gpt-oss-120b 0.97; Qwen3-32B H100 0.94 | Qwen3-32B A100, Qwen3-14B, Qwen3-8B, Llama-3.1-8B, gemma-2 (x2), Mistral (4090 x2, L4), GPTQ, AWQ, Qwen3-0.6B-FP8 4090; Qwen3-0.6B-FP8 H100 0.76 (ambiguous) |

### 3. CUDA graphs (the 800 MB constant): -0.43 to +2.50 GiB

vLLM subtracts its estimate: it captures piecewise graphs in full and FULL decode graphs
for the two largest sizes only, then extrapolates `first + (n - 1) x second`
(`v1/worker/gpu/cudagraph_utils.py:718-831` / `847-960`).  Per decoder layer and FULL
decode graph (capture sizes up to max_num_seqs: 35 at 256 sequences, 51 at 1024):

| class | per layer-graph (MiB) |
|---|---|
| sm8x, dense attention | 0.334-0.394 (Qwen3 0.6B-32B, Llama, Mistral, gemma-2, GPTQ, TP 2) |
| sm8x, other | mamba-1 0.197-0.210; MLA (DeepSeek-V2-Lite) 0.628 |
| sm90 | gemma-4 0.254, Qwen3-0.6B-FP8 0.387, gpt-oss 0.446, Muse 0.514, AWQ 0.535, Qwen3.6 0.627, Qwen3-32B 0.687; MLA (GLM) 1.388 |

New rule: layers x FULL decode graphs x the class median (sm8x 374 KiB, sm90 537 KiB).
The sm8x dense boots agree to +-8%.  Not modelled, stays in the error: MLA (one point per
class; GLM +2.02 GiB), the sm90 spread (gemma-4 -0.81, Qwen3-32B +0.52).

### 4. Weights: 0.00 to +0.66 GiB (unchanged except NemotronH)

- **Fixed:** NemotronH's mapper drops the `mtp` prefix (`models/nemotron_h.py:711` /
  `:728`); Nemotron-3.5-Lightning stores 2.49 GiB (2,670,651,904 B) of `mtp.layers.*`
  that the planner counted.  Qwen3-Next drops `mtp.` too (`qwen3_next.py:808` / `:810`);
  both joined `_MTP_NAMES_DROPPED`.  (vLLM also drops it for colqwen3_5, ernie4_5_moe,
  exaone4_5, exaone_moe, minicpmv4_6 -- not added: their model_type strings are not
  checked here.)
- **Traced, not fixed:** gemma-4 +0.53: vLLM allocates the rotary cos/sin caches inside
  the weights profiler, [max_position, rotary_dim] in the model dtype per distinct rope
  config (`rotary_embedding/base.py:58-63`, `get_rope` dtype default): 262,144 x (256 +
  512) x 2 = 0.375 GiB for gemma-4's sliding and full layers; negligible (<= 32 MiB)
  elsewhere.  Open: gpt-oss +0.66 (MXFP4 expert layout on Hopper, untraced), Muse +0.37.
  All within the weight test's 6%.

### 5. Peak (activation): unchanged here

Rewritten separately (probe rule, then the v0.30 encoder moment): +0.00 to +0.05 GiB.

## After

`scripts/kv_budget_residuals.py`; "segment" is the compile segment the plan may hold.

| record | model | GPU | TP | stored | now | total | weights | peak | non-torch | graphs | segment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0078d8e3 | Qwen/Qwen3-0.6B-FP8 | H100 80GB HBM3 | 1 | +0.07 | +0.10 | +0.00 | +0.04 | +0.02 | +0.22 | -0.19 | 0.29 |
| 066392da | Qwen/Qwen3-1.7B | GeForce RTX 4090 | 1 | +0.11 | +0.63 | +0.00 | +0.02 | +0.00 | +0.62 | +0.00 | 0.58 |
| 0ac02bfd | state-spaces/mamba-2.8b-hf | GeForce RTX 4090 | 1 | +5.75 | +0.15 | +0.00 | -0.00 | +0.00 | +0.50 | -0.34 | 0.48 |
| 205b397c | Qwen/Qwen3-1.7B | L4 | 1 | +1.40 | +0.59 | +0.00 | +0.02 | +0.00 | +0.58 | -0.01 | 0.58 |
| 374d2a3c | Qwen/Qwen3-32B | A100 80GB PCIe | 1 | -4.10 | -0.01 | -0.01 | +0.01 | +0.00 | +0.01 | -0.03 | 1.45 |
| 3903ded9 | Qwen/Qwen3-14B | RTX A6000 | 1 | -2.13 | -0.01 | +0.00 | +0.01 | +0.00 | -0.02 | +0.00 | 1.45 |
| 46447ffc | Qwen/Qwen3-8B | GeForce RTX 4090 | 1 | -0.87 | -0.00 | -0.00 | +0.01 | -0.00 | -0.00 | -0.02 | 1.16 |
| 56e563ed | Qwen/Qwen3-1.7B | GeForce RTX 4090 | 1 | +0.11 | +0.63 | +0.00 | +0.02 | +0.00 | +0.62 | +0.00 | 0.58 |
| 636ca167 | meta-llama/Llama-3.1-8B-Instruct | GeForce RTX 4090 | 1 | -0.55 | +0.07 | +0.00 | +0.04 | +0.00 | +0.00 | +0.03 | 0.98 |
| 6810a074 | state-spaces/mamba-2.8b-hf | GeForce RTX 4090 | 1 | +0.13 | -0.05 | +0.00 | +0.07 | +0.00 | +0.24 | -0.37 | 0.24 |
| 697d834d | deepseek-ai/DeepSeek-V2-Lite | A100 80GB PCIe | 1 | +0.80 | +0.65 | -0.01 | +0.06 | +0.00 | +0.35 | +0.24 | 0.39 |
| 6b393c53 | google/gemma-2-2b-it | GeForce RTX 4090 | 1 | +0.22 | +0.15 | -0.00 | +0.07 | +0.00 | +0.04 | +0.03 | 1.10 |
| 75e50084 | mistralai/Mistral-7B-Instruct-v0.3 | GeForce RTX 4090 | 1 | -1.27 | +0.03 | +0.00 | +0.01 | -0.00 | -0.00 | +0.03 | 0.25 |
| 7b314f39 | google/gemma-2-2b-it | GeForce RTX 4090 | 1 | +0.69 | +0.15 | +0.00 | +0.07 | +0.00 | +0.04 | +0.03 | 1.10 |
| 8654b66b | Qwen/Qwen3.6-35B-A3B-FP8 | H100 80GB HBM3 | 1 | +2.47 | +0.49 | +0.00 | +0.14 | +0.04 | +0.09 | +0.21 | 0.00 |
| 89055d7f | zai-org/GLM-4.7-Flash | H100 80GB HBM3 | 1 | +3.45 | +2.77 | +0.00 | +0.10 | +0.01 | +0.64 | +2.02 | 0.59 |
| 9ac3de78 | Qwen/Qwen3-8B | A100-SXM4-80GB | 2 | -10.41 | -0.05 | -0.00 | +0.01 | +0.01 | -0.06 | +0.00 | 0.00 |
| ab41dd7b | meta-models/Muse-Glimmer-30B | H100 80GB HBM3 | 1 | +1.63 | +0.45 | +0.00 | +0.37 | +0.05 | +0.07 | -0.03 | 0.00 |
| ac02237c | mistralai/Mistral-7B-Instruct-v0.3 | L4 | 1 | +0.03 | -0.00 | +0.00 | +0.01 | -0.00 | -0.03 | +0.03 | 0.25 |
| b9e40943 | JunHowie/Qwen3-8B-GPTQ-Int4 | GeForce RTX 4090 | 1 | +0.59 | +0.03 | +0.00 | -0.00 | -0.00 | +0.03 | +0.01 | 1.16 |
| c078c3a7 | deepseek-ai/DeepSeek-V2-Lite-Chat | A100 80GB PCIe | 1 | +0.81 | +0.66 | +0.00 | +0.06 | +0.00 | +0.35 | +0.24 | 0.39 |
| c4ad8130 | mistralai/Mistral-7B-Instruct-v0.3 | GeForce RTX 4090 | 1 | -1.74 | +0.03 | -0.00 | +0.01 | -0.00 | -0.00 | +0.03 | 0.25 |
| c6220073 | Qwen/Qwen3-0.6B | GeForce RTX 4090 | 1 | +0.16 | +0.32 | +0.00 | +0.01 | +0.00 | +0.34 | -0.03 | 0.29 |
| ded0ff28 | openai/gpt-oss-120b | H100 80GB HBM3 | 1 | +3.15 | +1.57 | +0.00 | +0.66 | +0.00 | +1.05 | -0.14 | 1.08 |
| df650fa3 | Qwen/Qwen3-8B-AWQ | H100 80GB HBM3 | 1 | -0.03 | +0.16 | +0.00 | +0.14 | +0.01 | +0.00 | +0.02 | 1.16 |
| e3193888 | Qwen/Qwen3-0.6B-FP8 | GeForce RTX 4090 | 1 | -0.12 | +0.04 | +0.00 | +0.04 | +0.00 | +0.01 | +0.00 | 0.29 |
| ea39d742 | google/gemma-4-31B-it | H100 80GB HBM3 | 1 | +0.97 | -0.26 | +0.00 | +0.53 | +0.01 | -0.00 | -0.81 | 0.00 |
| eef514a1 | Qwen/Qwen3-32B | H100 80GB HBM3 | 1 | -0.49 | +1.89 | +0.00 | +0.01 | +0.00 | +1.36 | +0.52 | 1.45 |

The non-torch column's large entries are the held compile segments (compare the last
column).

### Distribution (28 records, GiB)

| | min | max | median | mean abs | within +-0.5 | within +-1 |
|---|---|---|---|---|---|---|
| stored (claims at run time) | -10.41 | +5.75 | +0.12 | 1.58 | 10 | 17 |
| before (calculator this morning, plan as booted) | -0.20 | +4.45 | +0.47 | 1.10 | 15 | 18 |
| after | -0.26 | +2.77 | +0.15 | 0.43 | 20 | 25 |
| after, held segment set apart | -0.26 | +2.18 | +0.05 | 0.22 | 27 | 27 |

The one record past +-0.5 with the segment set apart is GLM-4.7-Flash (+2.18: MLA CUDA
graphs +2.02).

## The safety buffer

`KV_BUDGET_SAFETY_BUFFER_BYTES` (layered.py) = the largest over-prediction measured over
all 28 records with the plan's compile segment set apart, max(residual - segment):
2,340,479,996 B (GLM-4.7-Flash), + 0.005 GiB for the log's rounding of the measured
figure, rounded up to 0.01 GiB = **2.19 GiB** (2,351,494,595 B).  It replaces the 4 GiB
margin fitted to Qwen3.6.  The guard subtracts buffer + `compile_segment_bytes` of the
plan.  `test_the_safety_buffer_is_the_largest_measured_error` re-derives it from the
records and fails if any record exceeds it or if it exceeds the largest error by more than
the rounding; `test_kv_budget_prediction_matches_what_vllm_left` bounds every record at
`[-0.262 GiB, buffer + segment]` (under-prediction: gemma-4's -0.257 + rounding).

## Guard picks (H100, vLLM v0.30.0, the planner against the Hub, 2026-09-28)

| model | predicted KV at 1024 seqs | blocks | margin | max_num_seqs | vLLM |
|---|---|---|---|---|---|
| Qwen/Qwen3.8-27B | 16.12 GiB | 336 x 51,380,224 B | 2.19 (vision wrapper: no segment) | **291** (was 307) | failed at 1024 (v0.29 profiling cache, 512 blocks) |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | 9.51 GiB | 793 x 12,877,824 B | 2.19 + 0.66 segment | **556** (was 395) | 733 blocks measured |
| Qwen/Qwen3.6-35B-A3B-FP8 | 34.14 GiB | 1694 x 21,626,880 B | 2.19 | default 1024 kept | 1610 blocks measured |

- Nemotron: 793 predicted vs 733 measured: +60 blocks (0.72 GiB), inside the 0.66 GiB
  segment + buffer; with the MTP fix the prediction no longer lands near 733 by
  cancelling errors (before: 80 GiB nominal +0.74, MTP weights -2.49, non-torch/graphs
  under-counted).  556 <= 733.
- Qwen3.6: from the recorded inputs (processor fields included) 1634 blocks vs 1610
  (+0.49 GiB: weights +0.14, graphs +0.21, non-torch +0.09, peak +0.04).  The planner's
  own run predicts 1694 because `run_plan_pipeline` does not read the processor config,
  so its activation lacks the encoder moment (0.66 GiB instead of 1.92) -- open, see below.

## Open

- **The planner does not feed the encoder moment.**  `_activation_estimate` needs the
  processor fields (`activation_config(config, processor)`); `run_plan_pipeline` passes
  the bare config, so vision wrappers (Qwen3.6, Qwen3.8, Muse) are planned ~1.2 GiB high.
  The KV test uses the recorded fields, so it does not see this.
- MLA CUDA-graph estimate (GLM-4.7-Flash 3.25 GiB): sets the buffer.  One MLA point per
  GPU class; needs a second MLA boot per class or a trace of the FULL-graph samples.
- Weights: gemma-4 rope caches (traced, 0.375 GiB), gpt-oss +0.66, Muse +0.37.
- Base and graph costs are keyed on compute capability; the records confound it with the
  batch tier.  H200, B200, L40, A5000, 3090 totals are not detected yet.
- The held compile segment: allocator placement, not predicted; a warm compile cache
  avoids the benchmark (and moment 1) altogether -- untested.
