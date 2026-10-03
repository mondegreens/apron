# vLLM v0.29.0 hybrid memory: source trace (read-only, 2026-09-27)

Paths are under `.sources/vllm/vllm/` at 98dff2a.  Traced by a read-only agent;
the figures are predictions until a boot confirms them.

## Layer types

- Qwen3.5 family: `text_config.layer_types`, else built from
  `full_attention_interval` (default 4): layer i is linear unless
  `(i+1) % interval == 0` (`transformers_utils/configs/qwen3_5.py:88-97`,
  `qwen3_5_moe.py:94-101`); read per layer at `model_executor/models/qwen3_5.py:253`.
  State shapes come from `hf_text_config` (`qwen3_5.py:399-417, 619-637`).
- NemotronH: one layer per character of `hybrid_override_pattern`
  (`nemotron_h.py:572,589`): `M` Mamba2, `*` attention, `E` MoE, `-` MLP
  (`:525-530`).  Only `M` and `*` hold cache.

| Model | Cache-holding layers | Other |
|---|---|---|
| Qwen3.8-27B | 48 GDN + 16 full attention | — |
| Qwen3.6-35B-A3B-FP8 | 30 GDN + 10 full attention | — |
| Nemotron-3-Nano-30B NVFP4 | 23 Mamba2 + 6 attention | 23 MoE |

MTP layers add cache only with speculative decoding.

## State per layer per sequence

- GDN (`layers/mamba/mamba_utils.py:258-279`): `conv_dim = 2*k_heads*k_dim + v_heads*v_dim`;
  conv `(K-1+spec, conv_dim/TP)`; temporal `(v_heads/TP, v_dim, k_dim)`.
- Mamba2 (`mamba_utils.py:184-209`): `ng = n_groups (+ TP - n_groups if TP does not divide it)`
  (`:246-255`); conv `(K-1+spec, (heads*head_dim + 2*ng*state)/TP)`;
  temporal `(heads/TP, head_dim, state)`.
- Dtypes (`mamba_utils.py:97-109`): conv = `mamba_cache_dtype` ("auto" = model
  dtype); temporal = `mamba_ssm_cache_dtype` if set.  Qwen3.5 sets it from
  `text_config.mamba_ssm_dtype` = float32 (`model_executor/models/config.py:801-825`);
  Nemotron from `mamba_ssm_cache_dtype`, float32 by default (`:660-687`).
- Bytes: `MambaSpec.page_size_bytes = sum(prod(shape) * dtype_size)` (`v1/kv_cache_interface.py:873-881`).

| Model (TP 1) | conv | temporal | per layer |
|---|---|---|---|
| Qwen3.8-27B | 3*10240*2 = 61,440 | 48*128*128*4 = 3,145,728 | 3,207,168 |
| Qwen3.6-35B | 3*8192*2 = 49,152 | 32*128*128*4 = 2,097,152 | 2,146,304 |
| Nemotron | 3*6144*2 = 36,864 | 64*64*128*4 = 2,097,152 | 2,134,016 |

Attention KV per token per layer = `kv_heads/TP * (head + head_v) * dtype`
(`kv_cache_interface.py:406-416`): Qwen3.8 4,096; Qwen3.6 2,048; Nemotron
1,024 at bf16 (512 at fp8).

## Allocation

- Prefix caching is on by default for hybrids (`config/model.py:2128-2130`,
  `engine/arg_utils.py:2766,2796-2797`) → `mamba_cache_mode = "align"`
  (`model_executor/models/config.py:622-649`); off → `"none"`, `mamba_block_size = max_model_len` (`:650-657`).
- Page unification (`platforms/interface.py:767-947`, called at `:644-645`):
  `A` = attention bytes/token/layer, `P` = mamba page; `align = max(kernel min block, 16)` (`:883-889`);
  for none/align `attn_block = align * ceil(P / (align*A))` (`:911-914`); block size
  raised to it (`:916-917`); align mode sets `mamba_block_size = block_size` (`:924-925`);
  mamba page padded to `block_size*A` (`:928-938`, `layers/mamba/abstract.py:70-75`).
- Grouping (`v1/core/kv_cache_utils.py:1205-1345`): `group_size = min(#attn, #mamba)`
  (max if `max < 1.5*min`, `:1310-1321`); each type split into `ceil(n/group_size)` groups;
  `bytes_per_block = group_size * page` (`:1358-1370`); `num_blocks = available // bytes_per_block` (`:1440`).
- Blocks per request at `L = max_model_len` (`kv_cache_utils.py:1047-1069`): attention
  group `ceil(L/block_size)` (`kv_cache_interface.py:464-469`); each mamba group 2 in
  align, 1 in none (`:883-895`), plus speculative blocks.
- `max_concurrency = num_blocks / blocks_per_req`; `GPU KV cache size tokens = int(max_concurrency * L)`
  (`kv_cache_utils.py:2011-2038`).  **Bytes per sequence = blocks_per_req * group_size * page.**

| L = 262,144, TP 1, align, 16-token alignment | block | page | groups | bytes/block | blocks/req | bytes/sequence |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | 784 | 3,211,264 | 1 + 3 (16) | 51,380,224 | 335 + 6 | 17,520,656,384 (16.32 GiB) |
| Qwen3.6-35B | 1056 | 2,162,688 | 1 + 3 (10) | 21,626,880 | 249 + 6 | 5,514,854,400 (5.14 GiB) |
| Nemotron, bf16 KV | 2096 | 2,146,304 | 1 + 4 (6) | 12,877,824 | 126 + 8 | 1,725,628,416 (1.61 GiB) |

Prefix caching off: 17,366,515,712 / 5,449,973,760 / 1,674,117,120 bytes.

## Padding and caveats

- Mamba page padding +0.128% / +0.763% / +0.576% (logged "Padding mamba page size by X%").
- Attention rounds L up to whole blocks.
- Blackwell with FA4 and head_dim 256: alignment 128 (`v1/attention/backends/flash_attn.py:93-116`, `fa_utils.py:14`).
- Qwen3.5 checkpoints include a vision tower; it is loaded unless every modality
  limit is 0 (`--language-model-only` or `--limit-mm-per-prompt` 0) (`models/interfaces.py:365-376`,
  `config/multimodal.py:101,495-496`, `engine/arg_utils.py:1368`).
- Nemotron NVFP4: `hf_quant_config.json` `kv_cache_quant_algo: FP8` most likely makes the
  KV cache fp8 (`transformers_utils/config.py:767-775`, `utils/torch_utils.py:450-485,506-524`,
  `arg_utils.py:2070`): block 4176, page 2,138,112, 63 + 8 blocks x 12,828,672 =
  910,835,712 bytes (0.85 GiB) per sequence.  Runtime order not confirmed.
- Not traced: backend selection per platform; TP>1 `n_groups` rounding; the null block;
  encoder profiling memory.
