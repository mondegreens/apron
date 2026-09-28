# GLM-5.3 (glm_moe_dsa) and MiniMax-M3: KV pages and weights in vLLM v0.30.0 (trace, 2026-09-28)

Source read at `.sources/vllm-v0.30.0`, no boot.  vLLM's own `get_kv_cache_groups`
and the functions it dispatches to, `_get_kv_cache_bytes_per_block` and
`_max_memory_usage_bytes_from_groups` (`v1/core/kv_cache_utils.py`) were executed
verbatim on the specs the model code builds; `layered.py` agreed with them in 192
of 192 cases (TP ∈ {1, 2, 4, 8} × L ∈ {640, 4K, 32K, 128K, 256K, 1M} × in-flight ∈
{2K, 8K, 16K, 32K}, both models).  Scripts: `_thoughts/phase-1b-cohort/traces/glm53-minimax-m3/`
(outside the repo).  Headers: `tests/fixtures/cohort/headers/` (GLM-5.3 @aca966e4,
MiniMax-M3 @f0e1c1e0).

## Bytes per sequence per GPU (H200, SM90, bf16 KV, in-flight 16,384)

| Model | block | groups | bytes/block | L = 1,048,576 | L = 262,144 |
|---|---|---|---|---|---|
| GLM-5.3 (any TP) | 64 | 1 | 5,928,192 | 97,127,497,728 (90.46 GiB) | 24,281,874,432 (22.61 GiB) |
| MiniMax-M3 TP 8 (= TP 4) | 128 | 1 | 5,799,936 | 47,513,075,712 (44.25 GiB) | 11,878,268,928 (11.06 GiB) |

Both are full attention only: in-flight tokens do not change the figure.  Plus one
null block per GPU.

## Why they were unknown, and what vLLM does

- Both take the *uniform-type* path: every spec is a `FullAttentionSpec` subclass
  on one block size, so all layers form one `UniformTypeKVCacheSpecs` group whose
  block holds every layer's page (`kv_cache_utils.py:2282-2286`,
  `kv_cache_interface.py:1204-1206, 1233-1257, 1267-1271`).  New layout
  `"uniform"` in `layered.py`.
- GLM-5.3 runs `models/deepseek_v32` (`registry.py:122`).  KV `auto` stays bf16:
  SM90 picks FLASH_ATTN_MLA_SPARSE (FlashMLA sparse / FlashInfer SM90 next, same
  page) (`platforms/cuda.py:137-153`, `mla_attention.py:359-376`); latent
  512 + 64 = 1152 B/token on all 78 layers, one head, not divided by TP
  (`mla_attention.py:1347-1374`).  Indexer fp8 keys 128 + 4 = 132 B/token on the
  21 layers that pick their own top-k (`deepseek_v32/attention.py:166-200`; the
  rule matches the config's `indexer_types`).  Block 64 (all sparse MLA backends
  and the indexer backend).
- MiniMax-M3 (`minimax_m3_vl`): the ForConditionalGeneration suffix made the
  classifier say encoder-decoder; it is a decoder with a vision tower.  Every
  layer holds K + V of the KV heads (4 heads, TP 8: one per rank, `model.py:420-425`);
  the 57 sparse layers add one bf16 index key of 128 per token, a single head
  every rank keeps (`common/indexer.py:145-153`, `linear.py:1362-1367`).
- **MiniMax-M3 does not boot at vLLM's own block size.**  vLLM takes it from the
  first registered layer, a dense FLASH_ATTN layer (16); the sparse and index
  backends only run 128 (`sparse_attention.py:161-164`, `indexer.py:93-95`), and
  the worker's `select_common_block_size` (run verbatim) raises "No common block
  size for 16".  The plan now sets `block_size: 128` (`required_block_size`),
  as vLLM's recipe does ("mandatory on every platform").

## Weights per GPU at TP 8

| | GLM-5.3 | MiniMax-M3 |
|---|---|---|
| stored | 755,617,140,416 (fp8 block) | 854,172,958,720 (bf16) |
| MTP not loaded | 10,033,419,200 (layer 78) | 0 (no `mtp.*` stored) |
| whole on every rank | 1,713,656,448 | 274,067,456 |
| split by KV head (÷4, not ÷8) | – | 1,113,587,712 |
| **per GPU** | **94,699,576,704 (88.20 GiB)** | **107,150,251,008 (99.79 GiB)** |

- GLM whole per rank: fused q_a + kv_a (disable_tp), the indexer (wq_b, wk +
  weights_proj, k_norm), router gates, norms.  The indexer's fp8 `wk` is
  dequantized into a bf16 parameter and its scale dropped (`deepseek_v2.py:850-907`).
- M3 whole per rank: index_k, the float32 router gates (not downcast,
  `model.py:214-235`), norms, the vision patch embedding and row-parallel biases.
  The vision tower (1.61 GiB loaded) is split by TP and counted; it is loaded
  unless the plan says `--language-model-only`.
- Not counted (evidence against or too small): MLA's bf16 `W_UV` / `W_UK_T` copies
  (0.27 GiB/GPU if additive, but GLM-4.7-Flash's measured weights exceed the
  prediction by only 0.10 GiB against 0.40 GiB of such copies), the top-k index
  buffers (64 MiB GLM, 0.5 MiB M3).

## Planned (vLLM v0.30.0 H200 defaults: 8192 tokens, 1024 sequences; max_model_len 640)

| | weights | activation | total per GPU | fits 0.90 × 150.75 GB = 126.36 GiB | KV left | one sequence fits up to |
|---|---|---|---|---|---|---|
| GLM-5.3 | 88.20 | 2.77 | 94.15 GiB | yes | 32.44 GiB | 375,936 tokens (350,592 after the 2.19 GiB buffer) |
| MiniMax-M3 | 99.79 | 1.05 | 103.72 GiB | yes | 22.75 GiB | 539,008 tokens (487,168) |

Neither holds one sequence at its default 1,048,576 context: a boot without
`max_model_len` would fail vLLM's check (the planner sets 640).  Host RAM: none
(no Engram / PLE tables).

## Reasoning knobs (proposal, not implemented)

- M3's template reads `thinking_mode` ("enabled" / "disabled" / "adaptive";
  default adaptive = may think).  Suite v2's minimal reasoning:
  `chat_template_kwargs = {"thinking_mode": "disabled"}` (the prompt then ends in
  `</mm:think>`).  Its parser is registered as `minimax_m3`, the config says
  `minimax_m3_vl`, so `reasoning_parser_for` finds none.  Name it
  (`--reasoning-parser minimax_m3`, alias `minimax_m3_vl -> minimax_m3`): with
  thinking disabled it passes the output through as content and drops a leading
  stray `</mm:think>` (`reasoning/minimax_m3_reasoning_parser.py:170-195`).
- GLM-5.3's template always opens `<think>` and reads `reasoning_effort`
  (low / high, default max): `reasoning_request_fields` already sends "low", but
  `glm_moe_dsa` has no parser of that name, so reasoning lands in content
  (`glm45` / `glm47` are the registered GLM parsers).

## Inferred, to confirm on first boot

- GLM's backend is FLASH_ATTN_MLA_SPARSE only if the FA3 build is present; the
  alternatives use the same page.  Check "Using ... backend" and the block size.
- Activation: the sparse-MLA prefill path and M3's vision encoder peak are not
  traced (the plan notes the tower).  CUDA-graph cost uses the per-layer constant.
- SM100 / SM120, fp8 KV, `--language-model-only` not simulated.
