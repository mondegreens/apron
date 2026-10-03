# DeepSeek V4-Flash-0731 and V4.1-Flash: KV pages in vLLM v0.30.0 (trace, 2026-09-28)

Source read at `.sources/vllm-v0.30.0` (no boot). vLLM's own grouping
functions (`v1/core/kv_cache_utils.py` 1042-1100, 1113-1120, 1578-1611,
1943-2112, 2405-2461) were executed verbatim on the traced specs; the proposed
calculator sketch agreed with them in 40 of 40 cases
(L ∈ {4K, 32K, 128K, 256K, 1M} × in-flight ∈ {2K, 8K, 16K, 32K}).
Scripts: `_thoughts/phase-1b-cohort/traces/dsv4/` (outside the repo; scratch).  Per-tensor header dumps were
regenerated from safetensors headers and are not kept (17 MB).

## Facts that change Apron's inputs

- KV dtype is forced to `fp8_ds_mla` (uint8); bf16 KV is refused
  (`models/deepseek_v4/attention.py:94-123`, `deepseek_v41/attention.py:123-152`).
  Record 584 B per state, pages aligned to 576 B.
- KV is replicated on every TP rank (`num_kv_heads=1`, no TP division); per-GPU
  cache bytes are the same at TP 1, 4, 8.  DCP is refused for sliding specs.
- Grouping goes through the *packed* path (`_get_packed_kv_cache_groups`,
  `kv_cache_utils.py:1978-2112`) because the indexer backend only declares
  block-outermost layouts (`v1/attention/backends/mla/indexer.py:245-248`).
  `layered.py`'s group_size rule does not describe it.
- Default context 1,048,576 for both (YaRN factor not applied,
  `config/model.py:2468-2481`).
- MTP / DSpark layers are not built or loaded without a speculative config;
  checkpoint tensors are named `mtp.N.*`, which `plan_pipeline.mtp_tensor_bytes`
  (matches `layers.N.` only) does not subtract: +10.86 GB (V4), +7.93 GB (V4.1).
- V4.1 Engram tables (`layers.{1,14}.engram.*`, 202.8 GB) live in pinned host
  memory split across TP ranks (`config/engram.py:26-47`): ~189 GiB host RAM at
  TP 4, not GPU memory.
- V4.1 vision tower (0.97 GB) loads unless the image limit is 0.
- `gate.bias_vl` is named by v0.30 (`deepseek_v4/nvidia/model.py:871-878`), not
  by v0.29: V4.1 cannot load on v0.29 at all.

## Bytes per sequence per GPU (H200, SM90, in-flight 16,384 tokens)

| Model | block | groups | bytes/block | L = 1,048,576 | L = 262,144 |
|---|---|---|---|---|---|
| V4-Flash-0731 | 256 | 5 | 1,002,240 | 10,802,142,720 | 7,723,261,440 |
| V4.1-Flash | 64 | 9 | 116,928 | 2,339,027,712 | 902,216,448 |

In-flight 4,096: V4 5,798,960,640 / 2,720,079,360; V4.1 2,024,725,248 /
587,913,984.  Plus one null block per GPU.

## Inferred, to confirm on first boot

- V4: the ratio-4 compressor state (sliding spec, window 8, block 4) admits
  4,099 blocks per request at F=16,384 and dominates the total.  Check against
  vLLM's "Maximum concurrency" log line.
- Replicated weights divided by TP today (fused_wqa_wkv, compressor, indexer,
  router gate): ~1.2 GiB (V4), ~1.0 GiB (V4.1) per GPU under-counted.
- V4.1 Marlin MXFP4 pads per-rank intermediate 576 → 640 at TP 4 on SM90:
  +7.47 GiB per GPU (Marlin's support check not traced).
- SM100 / SM120 variants not simulated.
