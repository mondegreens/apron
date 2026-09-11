# Architecture dispatch study — why the calculator is mechanism-specific, not model-specific

**Date:** 2026-09-07 · **Source:** adversarial proof challenge against vLLM source at pinned commit a1541f57
**Authority:** ADR-003 (prediction versus GPU-executed evidence boundary)

## Corrected claim

Apron must own a GPU-free resource calculator because vLLM's complete physical KV specification is obtained from instantiated attention layers, and its non-KV memory budget is profiled during GPU execution. The calculator is not one hand-written function per model. It normalizes immutable artifact metadata and dispatches to a bounded set of memory mechanisms, using pinned vLLM source as a versioned specification and GPU-executed vLLM as the conformance oracle.

## The pipeline (same for every model)

```
HuggingFace config.json + safetensors metadata
  → apron normalizes architecture, artifact and cache geometry
    (num_kv_heads, head_size, hidden_size, sliding_window, is_deepseek_mla, num_experts, ...)
  → ModelConfig sets flags from config + registry
    (use_mla, is_hybrid, is_attention_free, has_inner_state, get_sliding_window)
  → apron selects a mechanism calculator and emits a labeled prediction
  → legacy deployment records may seed the candidate and uncertainty
  → GPU vLLM instantiates layers and emits actual KV specs
  → GPU profiling measures activations, graphs and remaining KV budget
  → prediction and measurement are stored side by side
```

No per-model calculator is required, but architecture/mechanism code is. Configuration fields can select and parameterize many common predictors; they do not prove which modules and backend customizations the engine will actually instantiate. Unsupported or ambiguous dispatch returns `unknown` until a mechanism is implemented and tested.

## Proof by five maximally different architectures

### 1. Dense GQA (Llama-3.1-8B) → FullAttentionSpec

**config.json fields used:** `num_key_value_heads=8`, `num_attention_heads=32`, `hidden_size=4096`, `num_hidden_layers=32`

**Convertor path:** `get_total_num_kv_heads()` at `model_arch_config_convertor.py:147-163` scans 5 standard key names: `n_head_kv`, `num_kv_heads`, `num_key_value_heads`, `multi_query_group_num`, `num_attention_groups`. Finds `num_key_value_heads=8`.

**KV spec:** No `sliding_window` in config. `is_deepseek_mla` is false (`model_type` not in the MLA list). → `FullAttentionSpec` at `kv_cache_interface.py:447`.

**Formula:** `page_size = num_kv_heads × block_size × (head_size + head_size_v) × dtype_bytes`. For GQA with 8 KV heads vs 32 attention heads, KV cache is 4× smaller per layer than MHA — the formula handles this from the config value, no special case needed.

### 2. Multi-Latent Attention — MLA (DeepSeek-V3) → MLAAttentionSpec

**config.json fields used:** `model_type="deepseek_v3"`, `kv_lora_rank=512`

**Convertor path:** `is_deepseek_mla()` at `model_arch_config_convertor.py:311-339` checks `model_type` against a list of 20+ MLA model types: `deepseek_v2`, `deepseek_v3`, `deepseek_v4`, `kimi_k2`, `glm_moe_dsa`, `pangu_ultra_moe`, etc. One string match covers all MLA variants.

**KV spec:** `MLAAttentionSpec` at `kv_cache_interface.py:554`. `head_size_v=0` (single latent vector, no separate V projection). Completely different memory formula from FullAttention.

**Formula:** `page_size = num_kv_heads × block_size × kv_lora_rank × dtype_bytes`. NOT `hidden_size × 2` — the compressed latent representation is dramatically smaller. llmcalc's single KV formula (`2 × batch × seq × hidden × layers × bytes`) would be wrong by the compression ratio.

### 3. Sliding Window (Mistral-7B) → SlidingWindowSpec

**config.json fields used:** `sliding_window=4096`, `num_key_value_heads=8`

**Convertor path:** `ModelConfig.get_sliding_window()` at `model.py:1489-1491` reads `getattr(hf_text_config, "sliding_window", None)`. Returns 4096.

**KV spec:** `SlidingWindowSpec` at `kv_cache_interface.py:707`. KV cache is bounded by window size, not full sequence length.

**Formula:** `effective_tokens = min(sliding_window + extra, max_model_len)`, then `max_blocks = ceil(effective_tokens / block_size)`. At 4096 window and 32768 max_model_len, the KV cache is 8× smaller than FullAttention would compute — because only the last 4096 tokens need KV entries.

### 4. Hybrid Mamba+Attention (Jamba) → MambaSpec + FullAttentionSpec per layer

**config.json fields used:** `layers_block_type=["attention", "mamba", "attention", "mamba", ...]`

**Convertor path:** `get_num_layers_by_block_type()` at `model.py:1617-1656` reads the per-layer type list from `hf_text_config.layers_block_type`. Each layer is classified as attention or Mamba from the config, not by instantiating the model.

**KV spec:** Attention layers → `FullAttentionSpec`. Mamba layers → `MambaSpec` at `kv_cache_interface.py:885`. Each layer gets its own spec — the total KV is the sum of per-layer specs, not one formula applied uniformly.

**Formula for Mamba layers:** `page_size = sum(product(shape) × dtype_size for each state tensor)`. Completely different from attention — recurrent state, not key-value pairs. A single-formula calculator would be wrong for every Mamba layer.

### 5. MoE (Qwen3-30B-A3B) → FullAttentionSpec (KV) + different weight memory

**config.json fields used:** `num_experts=128`, `num_experts_per_tok=8`

**Convertor path:** `get_num_experts()` at `model_arch_config_convertor.py:186` reads from config.

**KV spec:** MoE does not change KV cache — experts affect weight memory and compute, not attention cache. → `FullAttentionSpec` for all attention layers.

**Why it matters for planning:** Weight memory calculation must account for MoE. All 128 experts' weights are loaded, but only 8 are active per token. The safetensors index includes ALL expert weights. The planner sums all tensor bytes from the index — correct automatically, because the index lists what's actually stored.

## Additional traces: Kimi, Gemma, GLM families

### 6. Kimi-K2 / Kimi-K3 (Moonshot AI) → MLAAttentionSpec (MLA, same as DeepSeek)

**config.json fields used:** `model_type="kimi_k2"` or `"kimi_linear"`

**Convertor path:** `is_deepseek_mla()` at `model_arch_config_convertor.py:328-329` includes `"kimi_k2"` and `"kimi_linear"` in the MLA model-type list alongside DeepSeek variants. One string match.

**Registry:** `KimiLinearForCausalLM` maps to `vllm.models.kimi_k3` module (registry.py:146-148). `KimiK25ForConditionalGeneration` maps to `kimi_k25` (registry.py:467). `KimiK3ForConditionalGeneration` maps to `vllm.models.kimi_k3` (registry.py:469). Speculative decoding: `DSparkKimiK3DraftModel` maps to `vllm.models.kimi_k3.nvidia.dspark_mla` (registry.py:638).

**KV spec:** Same as DeepSeek-V3 → **MLAAttentionSpec** with compressed latent KV. No per-Kimi calculator needed — the MLA detection covers all MLA-family models (DeepSeek, Kimi, GLM-MoE-DSA, Pangu, Bailing) from one list.

**Why this matters:** Kimi-K2.5 is one of the most popular commercial LLMs in China. It uses DeepSeek's MLA architecture under license. The same `is_deepseek_mla()` check at convertor.py:311-339 handles DeepSeek V2/V3/V4, Kimi K2/K3, GLM-MoE-DSA, Pangu Ultra MoE, and Bailing Hybrid — 20+ model_type strings, one code path, one KV spec class.

### 7. Gemma 3 (Google) → mixed FullAttentionSpec + RSWASpec per layer

**config.json fields used:** `layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention", ...]`, `sliding_window=4096`, `rswa_window` (Rotating Sliding Window Attention)

**Convertor path:** `rswa_window` read at `model_arch_config_convertor.py:385-386` from `hf_config`. Per-layer type from `config.layer_types` (same mechanism as hybrid Jamba, model.py:1636).

**Model implementation:** `gemma3.py:155` — each layer checks `self.is_sliding` from `config.layer_types[layer_idx]`. Sliding layers use `sliding_window` from config, full-attention layers don't.

**KV spec:** Full-attention layers → **FullAttentionSpec**. Sliding layers → **RSWASpec** at `kv_cache_interface.py:625` (extends FullAttentionSpec with rotation). The `RSWASpec` class is Gemma 3's unique contribution to the KV spec taxonomy — it's one of the 14 spec classes, handling Gemma's specific attention pattern.

**Config-only path:** `layer_types` is in config.json. `sliding_window` is in config.json. `rswa_window` is in config.json. All config-derivable. No model instantiation needed.

### 8. Gemma 4 (Google) → per-layer sliding/full with MoE + KV sharing

**config.json fields used:** `layer_types=["sliding_attention", "full_attention", ...]`, `sliding_window`, `rope_parameters` per layer type

**Model implementation:** `gemma4.py:444-445` — `layer_type = config.layer_types[layer_idx]`, `self.is_sliding = layer_type == "sliding_attention"`. Same config-driven per-layer dispatch as Gemma 3.

**New feature in Gemma 4:** KV cache sharing between layers (gemma4.py:476-479). Layers can share KV with a previous layer of the same type, reducing total KV memory. This is handled by `kv_sharing_target_layer_name` in the attention module, which the KV spec dispatch at `gpu_model_runner.py:7648-7658` skips (shared layers don't get their own spec — they reuse the target layer's cache).

**Config-only path:** All driven by `config.layer_types` and `config.sliding_window`. KV sharing is determined by `first_kv_shared_layer_idx` in config (gemma4.py:476). Config-derivable.

### 9. Gemma 3n (mobile, hybrid attention) → mixed FullAttention + SlidingWindow per layer

**config.json fields used:** `layer_types=["full_attention", "sliding_attention", ...]`, `sliding_window`

**Model implementation:** `gemma3n.py:333-335` — same pattern: `layer_type = config.layer_types[layer_idx]`, `is_sliding = layer_type == "sliding_attention"`. Despite being a mobile-optimized model with novel architecture (GatedDeltaNet linear attention layers), the KV-relevant attention layers follow the standard sliding/full split.

**Config-only path:** Same as Gemma 3/4. All from `config.layer_types` and `config.sliding_window`.

### 10. GLM-4 MoE Lite (Zhipu AI / THUDM) → MLAAttentionSpec (reuses DeepSeek MLA)

**config.json fields used:** `model_type="glm4_moe_lite"` (in the MLA detection list at convertor.py:326)

**Registry:** `Glm4MoeLiteForCausalLM` maps to `glm4_moe_lite` module (registry.py:121). And `GlmMoeDsaForCausalLM` maps directly to `vllm.models.deepseek_v32` (registry.py:122) — GLM MoE DSA literally reuses DeepSeek V3.2's implementation.

**Model implementation:** `glm4_moe_lite.py:59-61` imports `DeepseekV2MLAAttention` directly: `from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention`. The GLM-4 MoE attention class (`Glm4MoeLiteMLAAttention` at line 100) inherits from `DeepseekV2MLAAttention`.

**KV spec:** Same as DeepSeek → **MLAAttentionSpec**. Same formula, same compressed latent, same code. GLM literally subclasses DeepSeek's attention.

**Why this matters:** Zhipu AI's GLM-4 MoE reuses DeepSeek's attention implementation. That is evidence for a shared MLA mechanism calculator instead of a per-model GLM calculator. GPU conformance still establishes the backend-customized physical spec for a particular engine build.

## Summary: 10 architectures, shared mechanisms, no per-model formulas

| # | model | KV spec class | config fields that determine it |
|---|---|---|---|
| 1 | Llama-3.1-8B (dense GQA) | FullAttentionSpec | `num_key_value_heads` |
| 2 | DeepSeek-V3 (MLA) | MLAAttentionSpec | `model_type` in MLA list + `kv_lora_rank` |
| 3 | Mistral-7B (sliding window) | SlidingWindowSpec | `sliding_window` |
| 4 | Jamba (hybrid Mamba+attention) | MambaSpec + FullAttentionSpec per layer | `layers_block_type` |
| 5 | Qwen3-30B-A3B (MoE) | FullAttentionSpec | `num_experts` (affects weights, not KV) |
| 6 | Kimi-K2/K3 (MLA) | MLAAttentionSpec | `model_type` in MLA list |
| 7 | Gemma 3 (RSWA) | FullAttentionSpec + RSWASpec per layer | `layer_types` + `rswa_window` |
| 8 | Gemma 4 (sliding + KV sharing + MoE) | FullAttention + SlidingWindow per layer, with sharing | `layer_types` + `sliding_window` |
| 9 | Gemma 3n (mobile hybrid) | FullAttention + SlidingWindow per layer | `layer_types` + `sliding_window` |
| 10 | GLM-4 MoE Lite (MLA, reuses DeepSeek) | MLAAttentionSpec | `model_type` in MLA list |

All 10 can share a small mechanism taxonomy. Much of the prediction input is config-derivable; the instantiated layer set, backend customization and profiled memory remain engine-executed evidence. The result is bounded architecture code rather than either extreme: neither 113 hand-coded model functions nor a fictitious universal config-only vLLM oracle.

## What the planner builds on top (not in vLLM)

| component | source | complexity |
|---|---|---|
| Weight bytes | safetensors index: sum per-tensor bytes from metadata headers, account for TP sharding | Metadata reading + TP division |
| Activation + overhead | prediction from a hardware/software/model/workload fingerprint, calibrated from matching records; GPU `profile_run()` supplies the measurement | Fingerprinted prediction plus measured calibration; never portable merely by model/config equality |
| VRAM budget | `total × gpu_memory_utilization - weights - activation - overhead` | Subtraction |
| Config → cache mechanism selection | Normalize explicit artifact fields into MHA/GQA/MQA, MLA, sliding/hybrid, recurrent-state or unsupported; do not default an unknown architecture to MHA | Versioned dispatch with explicit unknowns |
| TP/quant recommendation | Traverse the complete typed candidate graph: native/publisher/third-party artifacts, offline derivations and online transforms; use actual tensor/scale bytes, apply version-pinned constraints, and require evaluation-derived task evidence plus serving evidence for the exact endpoint before `Recommended` | Extensible adapters with explicit unsupported states and evidence authority |
| Combined verdict | Budget ≥ KV demand → fits; else → doesn't fit, with the binding constraint named | Comparison |

The planner implements prediction math by mechanism and tests it against pinned vLLM behavior. It does not duplicate model implementations. vLLM's spec classes, validators and registry are source inputs and GPU conformance targets, not proof that arbitrary models can be planned by executing vLLM on CPU.

## Implementation guidance

### Where to watch during development

1. **The 5-key scan in `get_total_num_kv_heads()`** (`model_arch_config_convertor.py:147-163`). If a model's config.json uses a key name outside the 5 known aliases, the convertor silently defaults to MHA (num_attention_heads). The planner should detect this fallback and flag the record as `explicit-unknown` per INV-10 rather than silently using the wrong number. One if-statement wrapper.

2. **The instantiated-layer boundary.** `gpu_model_runner.py:7633-7664` discovers attention modules from the static forward context and calls each module's `get_kv_cache_spec()`, then lets the selected backend customize it. The CPU predictor cannot assume that importing registry metadata reproduces this physical layer set. A meta-device probe is optional research, not the prediction architecture.

3. **Config resolution failures** (`transformers_utils/config.py:718-737`). Models without `config.json` or `params.json` (GGUF-only uploads, adapter-only LoRA repos) crash before the registry is ever reached. The planner's Step 1 Resolve must handle this with per-cause error messages: GGUF → "use llama.cpp --fit", adapter-only → "specify base model", unparseable → explicit error.

4. **Quantized KV cache byte packing** (`attention.py:628-638`, `customize_spec` at `gpu_model_runner.py:7662`). Even when standard BF16/FP16 prediction is directly derivable, quantized byte packing can depend on the selected attention backend and target platform. Quantized-KV configurations remain platform-dependent predictions until GPU execution.

5. **Transformers-backend models** (23 architectures in `_TRANSFORMERS_SUPPORTED_MODELS`). These use vLLM's own `Attention`/`MLAAttention` classes via `transformers/base.py:573` (`create_attention_instances`), so the KV math is the same. But the forward-pass kernel differs (HF module graph via `torch.fx`). The planner should note this in the record as a different performance class — same memory model, different throughput characteristics.

### What to build and what not to build

- Build cache calculators for supported memory mechanisms, not one formula per model.
- Build a normalized `ModelSpec` from immutable Hub/artifact metadata; do not maintain a lossy copy as the authoritative model catalogue.
- Extract versioned compatibility constraints from pinned engine source; do not create one timeless precision-to-GPU table.
- Return `unknown` when the artifact lacks the fields required by a mechanism; do not silently default to MHA or a percentage.
- Use the llmcalc deployment catalogue as evidence and candidate priors; do not confuse those observations with universal formulas.
- Use GPU-executed vLLM for physical KV spec, profiling, boot and kernel conformance; do not describe CPU source inspection as vLLM validation.

### Source references (all from vLLM at pinned commit a1541f57)

| component | file | key lines |
|---|---|---|
| Config field extraction | `transformers_utils/model_arch_config_convertor.py` | 52-71 (fields), 147-163 (KV heads), 311-339 (MLA detection) |
| KV spec classes (14) | `v1/kv_cache_interface.py` | 447-1073 |
| Feasibility check | `v1/core/kv_cache_utils.py` | 956 (`check_enough_kv_cache_memory`), 902 (`estimate_max_model_len`), 1048 (`get_max_concurrency`) |
| Validators (~268) | `config/model.py`, `config/vllm.py`, `config/scheduler.py` | 1417-1488 (TP/PP/DCP), 790-811 (dtype/quant vs compute capability) |
| Model registry | `model_executor/models/registry.py` | 72-754 (10 dicts), 1020-1067 (`inspect_model_cls` + JSON cache) |
| _ModelInfo flags | `model_executor/models/registry.py` | 865 (`from_model_cls`), 945-999 (JSON cache read) |
| Hybrid per-layer types | `config/model.py` | 1617-1656 (`get_num_layers_by_block_type`) |
| Sliding window | `config/model.py` | 1489-1491 (`get_sliding_window`) |
| Max model len | `config/model.py` | 2008 (`get_and_verify_max_len`) |
| NVIDIA aiconfigurator comparison | [`aiconfigurator`](https://github.com/ai-dynamo/aiconfigurator) `aic-core/src/aiconfigurator_core/sdk/backends/vllm_backend.py` | 19-20 (3.3× error acknowledged), 40-41 ("mirrors TRT-LLM's activation model"), 53-55 (reuses TRT-LLM coefficients) |

### Corrected llmcalc reuse finding

llmcalc contains two assets that must not be collapsed:

1. **Deployment-tested catalogue — carry forward.** The current registry has 113 unique model ids (101 non-Gemma and 12 Gemma; one duplicate textual key). Every entry carries `vram_fp16_gb`; the primary estimator uses it directly as the total from real-world testing (`resource_estimator.py:425-458`). Registry TP and vLLM overrides take priority (`resource_estimator.py:460-507`; `parameter_builder.py:521-569`). The owner confirms that all non-Gemma catalogue models were successfully deployed on RunPod. The deployment manager passes the generated environment to `runpod.create_pod` and records health and inference endpoints (`_implementation_instructions/source_code/runpod_vllm_manager.py:708-763, 869-941`). Import this as owner-attested legacy boot evidence and retain every surviving parameter.
2. **Generic fallback formulas — redesign.** The single hidden-size KV formula, fixed 10% activation estimate, negative MoE adjustment, generic quantization reduction and noisy GPU table do not generalize safely. They become historical baselines and regression cases, not the new prediction core.

The catalogue is a stronger seed for practical RunPod boot candidates than either vLLM alone or aiconfigurator supplies for this niche. Its limitation is reproducibility metadata: the surviving container setup pins vLLM 0.9.1 in requirements but uses a mutable `:latest` image name, and the registry does not preserve a complete model-revision, image-digest, GPU, workload and measurement record per attempt. Apron preserves that uncertainty, replays a stratified cohort, and graduates records without discarding the expensive experience that created them.
