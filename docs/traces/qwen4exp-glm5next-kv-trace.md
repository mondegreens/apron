# Qwen3.8-Flash-Next (Qwen4Exp) and GLM-5.3-Flash (Glm5Next): KV pages in vLLM v0.30.0 (trace, 2026-09-28)

Source read at `.sources/vllm-v0.30.0`, no boot.  Independent simulator and a
prototype `layered.py` extension in `_thoughts/phase-1b-cohort/traces/qwen4exp-glm5next/` (outside the repo); the prototype
reproduces the numbers below and still passes the Qwen3.5 / Nemotron / Gemma 4
expectations.

## Bytes per sequence per GPU (bf16, L = 262,144)

| Model | TP | block | bytes/block | blocks | bytes/sequence | prefix caching off |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 4 | 784 | 10,235,904 | 344 | 3,521,150,976 | 3,480,207,360 |
| Qwen3.8-Flash-Next | 1 | 1568 | 39,739,392 | 177 | 7,033,872,384 | 6,874,914,816 |
| GLM-5.3-Flash | 4 | 1152 | 13,394,304 | 237 | 3,174,450,048 | 3,120,872,832 |
| GLM-5.3-Flash | 1 | 4352 | 50,600,704 | 70 | 3,542,049,280 | 3,339,646,464 |

Boot-log checks: Qwen "attention block size to 784", mamba padding 0.13%,
BLNHC layout; GLM block 1152, padding 8.68%.

## Why the existing code does not fit

- Qwen4Exp is not Qwen3.5: QSA adds a compressed index key and a raw-key ring per
  full-attention layer, a TP-replicated PLE conv state, a block-outermost (packed)
  layout (`qsa_cache.py:753-756`), bf16-only KV.  Applying the qwen3_5 code with a
  model_type change gives -6.7% (TP4).
- GLM5Next has its own grouping path (`kv_cache_utils.py:1195-1275`): KDA state,
  MLA latent (NoPE, 512), fp8 indexer (tokens_per_state 4), a kpool tail.

## Weight prediction problems

- Qwen: PLE n-gram table 95.4 GiB lives in pinned host memory by default
  (`VLLM_PLE_CPU_OFFLOAD=1`), `mtp.*` (5.2 GB, keyed by `mtp_num_hidden_layers`)
  not subtracted, vision 0.84 GiB, QSA top-k buffers 0.75 GiB at 8192 tokens.
  Net ≈ 58.8 GiB/GPU at TP4 (Apron today: 83.8).
- GLM: MTP layer already subtracted; vision 1.05 GiB; net ≈ 74.7 GiB/GPU at TP4:
  does not fit TP4 on H100 (inference) — H200 or TP8.

## Not pinned

GLM sparse MLA backend depends on a runtime FlashInfer probe (same page at bf16);
speculative decoding not modelled; CUDA-graph memory; PLE host-RAM limits.
