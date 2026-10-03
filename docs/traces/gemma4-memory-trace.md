# vLLM v0.29.0 Gemma 4 memory: source trace (read-only, 2026-09-27)

Paths under `.sources/vllm/vllm/` at 98dff2a.  Traced by a read-only agent;
predictions until a boot confirms them.

## Per-layer attention (google/gemma-4-31B-it)

- Wrapper `model_executor/models/gemma4_mm.py` builds `Gemma4ForCausalLM` (:1126-1132);
  text model `model_executor/models/gemma4.py`; per-layer config
  `transformers_utils/configs/gemma4.py:10-34`.
- Full layers: `head_dim = global_head_dim` (512), `kv_heads = num_global_key_value_heads`
  (4) when `attention_k_eq_v` (configs/gemma4.py:26-33); sliding layers 256 / 16.
  Used at gemma4.py:576-585.  Window 1024 on sliding layers only (:445-446, :509).
- `attention_k_eq_v`: no `v_proj` in the checkpoint for full layers; vLLM copies
  `k_proj` into a `v_proj` slot (gemma4.py:1711-1716): ~210 MiB of extra bf16
  weights (10 x 4 x 512 x 5376 x 2).  K and V are both cached (k_norm+RoPE vs
  v_norm, :538-552), `head_size_v = head_size` (kv_cache_interface.py:397-399).
- Specs: sliding → `SlidingWindowSpec`, full → `FullAttentionSpec`
  (layers/attention/attention.py:610-655).  No KV sharing (gemma4.py:470).

## Pages, groups

- Bytes/token/layer `kv_heads * (hd_k + hd_v) * dtype` (kv_cache_interface.py:412,416):
  sliding 16,384; full 8,192.
- Block 16 (config/cache.py:79; triton_attn.py:307-308, flash_attn.py:112-116;
  attention.py:95-121,634).  Pages 262,144 / 131,072 → full layers rescaled to
  block 32 (`unify_kv_cache_spec_page_size`, kv_cache_utils.py:1180-1184); no waste.
- Groups (kv_cache_utils.py:1310-1344): 10 full + 50 sliding → group size 10:
  1 full + 5 sliding groups.  Bytes/block 2,621,440 (:1358-1370).
- Hybrid KV cache manager on by default on CUDA (config/vllm.py:1847-1849).
- Full attention: 81,920 B per context token; full blocks/request `cdiv(L,32)`
  (kv_cache_interface.py:464-469).  Sliding blocks per group per request
  `cdiv(min(1023 + F, L), 16) + 1`, x5 (:723-740), with
  `F = max_concurrent_batches * max_num_batched_tokens` (config/vllm.py:572-580),
  `max_concurrent_batches = 2` with async scheduling (vllm.py:563-568,1311).

## Boot-log numbers (kv_cache_utils.py:1061-1068, 2011-2037)

`B_req = sum cdiv(max_mem_bytes, page)`; `concurrency = num_blocks / B_req`;
"GPU KV cache size" = `int(concurrency * max_model_len)`.  Startup check
(:2319-2336): `available - 2,621,440 >= B_req * 2,621,440` or it raises.

| setup (bf16, TP 1) | L | B_req | min KV needed |
|---|---|---|---|
| A100/L40S (F=4096) | 262144 | 9797 | 23.92 GiB |
| same | 32768 | 2629 | 6.42 GiB |
| H100/H200 (F=16384) | 262144 | 13637 | 33.29 GiB |
| same | 32768 | 6469 | 15.79 GiB |
| >=160 GB GPU (F=32768) | 262144 | 18757 | 45.79 GiB |
| same | 32768 | 11269 | 27.51 GiB |

## Vision tower, other factors

- Vision/audio towers load unless every modality limit is 0
  (gemma4_mm.py:1086-1123; interfaces.py:360-377; `--language-model-only`,
  multimodal.py:495-496, arg_utils.py:1368).  ~1 GiB bf16 (estimate).
- Multimodal profile transient capped at `min(free/2, total/10)` (gemma4_mm.py:1260-1282).
- No per-layer input embeddings for the 31B (`hidden_size_per_layer_input = 0`);
  lm_head tied (gemma4.py:1570-1571); softcap 30 (:1575); dtype bf16
  (config/model.py:2268-2290).  Mixed head dims force FA4 where available on
  SM90/100/110, else TRITON_ATTN (model_executor/models/config.py:209-256).
- Text weights ≈ 57.4 GiB with the duplicated v_proj (estimate).
- **Consequence:** on an H100 at the default max_model_len (262,144) the KV
  minimum (33.3 GiB) does not fit beside ~58 GiB of weights: expect a
  "no room for the KV cache" failure unless max_model_len is reduced.
- Uncertain: Blackwell FA4 hd256 block size 128; KV layout, connectors, MTP.

## 26B-A4B (MoE) differences

30 layers, 5 full; hidden 2816; 8 sliding / 2 global KV heads; 128 experts,
top-8.  Sliding 8,192 B/token/layer; full 4,096 (20,480 B/token over 5 layers);
pages 131,072 / 65,536 → full block 32; six groups of 5; 655,360 B/block.
