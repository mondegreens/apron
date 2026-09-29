"""Mechanism-aware memory calculators.

Each registered branch handles a specific attention/state mechanism:
- autoregressive_decode: dense GQA/MHA/MQA (Llama, Qwen, Gemma, etc.)
- mla_decode: Multi-Latent Attention (DeepSeek-V2/V3/R1)
- ssm_decode: Mamba-1 state-space models (a fixed state per sequence)

MoE activation adjustment and sliding-window KV cap are applied
within each branch when the config signals them.
"""

from __future__ import annotations

import math
from typing import Any

from apron.domain.mechanisms import CalculatorInput, register_calculator

DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "int4": 1,
}


_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def _kv_dtype_bytes(metadata: dict[str, Any], inputs: CalculatorInput) -> int:
    """KV cache element size: the engine's ``kv_cache_dtype`` when set, else
    the model dtype (vLLM's "auto")."""
    kv_dtype = str(inputs.execution_spec_data.get("kv_cache_dtype", "auto"))
    if kv_dtype != "auto":
        return DTYPE_BYTES[kv_dtype]
    return DTYPE_BYTES.get(metadata.get("torch_dtype", "bfloat16"), 2)


# vLLM pads the vocabulary to a multiple of 64 (vocab_parallel_embedding.py:33).
VOCAB_PADDING = 64


#: Config fields the startup-peak estimate reads, from the merged config (the
#: language model's ``text_config`` over the top level, as the planner merges
#: it).  ``vision_config`` / ``audio_config`` only matter as present or absent.
ACTIVATION_FIELDS: tuple[str, ...] = (
    "hidden_size",
    "vocab_size",
    "intermediate_size",
    "num_attention_heads",
    "head_dim",
    "global_head_dim",
    "final_logit_softcapping",
    # MoE
    "n_routed_experts",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "experts_per_token",
    "moe_intermediate_size",
    # MLA
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    # NemotronH: the Mamba-2 mixer, and the experts' activation -- the field
    # only NemotronH declares; vLLM builds those experts without a gate.
    "mamba_num_heads",
    "mamba_head_dim",
    "n_groups",
    "ssm_state_size",
    "mlp_hidden_act",
    # MoE routing: vLLM refuses FlashInfer TRTLLM's routing without it.
    "norm_topk_prob",
    # GLM5Next's kpool sparse indexer (its profile-run workspace).
    "index_head_dim",
    "index_kpool",
)
#: Fields ``activation_config`` derives from config.json rather than copies.
#: ``fp8_block_experts``: a block-FP8 checkpoint (``fp8_block_quant``).
DERIVED_ACTIVATION_FIELDS: tuple[str, ...] = ("fp8_block_experts",)
_TOWER_FIELDS = ("vision_config", "audio_config")

# The MLA prefill workspace is capped at 64k tokens (mla_attention.py:2163-2178)
# and aligned to the KV block (mla_attention.py:1864-1876).  The MLA backends
# the cohort GPUs pick (FLASH_ATTN_MLA on SM90, TRITON_MLA on SM80 --
# platforms/cuda.py:135-142) accept any multiple of 16, so the block stays at
# CacheConfig.DEFAULT_BLOCK_SIZE (config/cache.py:79; interface.py:626-637).
_MLA_WORKSPACE_CAP = 64 * 1024
_MLA_BLOCK = 16


#: Vision towers whose encoder-phase peak is traced (``_encoder_peak_bytes``),
#: by the checkpoint's top-level ``model_type``, in vLLM v0.30.0.  The Qwen 3.5 /
#: 3.6 wrappers run the Qwen3-VL tower (qwen3_5.py:111 -> qwen3_vl.py).
_VISION_FAMILIES: dict[str, str] = {
    "qwen3_vl": "qwen3_vl",
    "qwen3_vl_moe": "qwen3_vl",
    "qwen3_5": "qwen3_vl",
    "qwen3_5_moe": "qwen3_vl",
    "muse_glimmer": "muse_glimmer",
}
#: The fields ``activation_config`` derives for a traced vision tower (from
#: ``vision_config`` and the processor files).
ENCODER_FIELDS: tuple[str, ...] = (
    "vision_family",
    "vision_hidden_size",
    "vision_patch_size",
    "vision_merge_size",
    "vision_temporal_patch_size",
    "image_longest_edge",
    "video_longest_edge",
    "max_image_tokens",
    "max_video_frame_tokens",
    "video_num_frames",
)


def activation_config(
    config: dict[str, Any], processor: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The part of a checkpoint's config.json the activation estimate reads:
    ``text_config`` merged over the top level, towers reduced to a flag.

    *processor* is the checkpoint's processor configuration as
    ``{"image_processor": {...}, "video_processor": {...}}`` -- the shape of
    processor_config.json, or preprocessor_config.json and
    video_preprocessor_config.json put under those keys.  With it, a traced
    vision tower (``_VISION_FAMILIES``) also gets the fields that size vLLM's
    dummy encoder batch (``ENCODER_FIELDS``)."""
    text = config.get("text_config")
    merged = {**config, **text} if isinstance(text, dict) else dict(config)
    fields = {k: merged[k] for k in ACTIVATION_FIELDS if merged.get(k) is not None}
    fields.update({k: True for k in _TOWER_FIELDS if config.get(k)})
    if fp8_block_quant(config):
        fields["fp8_block_experts"] = True
    family = _VISION_FAMILIES.get(str(config.get("model_type")))
    vision = config.get("vision_config")
    if family is None or not isinstance(vision, dict) or processor is None:
        return fields
    image = processor.get("image_processor") or {}
    video = processor.get("video_processor") or {}
    encoder = {
        "vision_family": family,
        "vision_hidden_size": vision.get("hidden_size"),
        "vision_patch_size": vision.get("patch_size"),
        "vision_merge_size": vision.get("spatial_merge_size") or vision.get("merge_size"),
        "vision_temporal_patch_size": vision.get("temporal_patch_size")
        or vision.get("patch_temporal"),
        "image_longest_edge": (image.get("size") or {}).get("longest_edge"),
        "video_longest_edge": (video.get("size") or {}).get("longest_edge"),
        "max_image_tokens": image.get("max_image_tokens"),
        "max_video_frame_tokens": video.get("max_video_frame_tokens"),
        "video_num_frames": video.get("num_frames"),
    }
    fields.update({k: v for k, v in encoder.items() if v is not None})
    return fields


def _qwen_vl_merged_tokens(max_pixels: int, unit: int) -> int:
    """Merged tokens of Qwen2-VL's image "with most features": the largest
    count <= max_pixels / unit^2 with a factor pair of aspect <= 200
    (qwen2_vl.py:969-1021; tokens = grid_t x h x w / merge^2, grid_t = 1)."""
    for count in range(max_pixels // (unit * unit), 0, -1):
        low = next(d for d in range(int(count**0.5), 0, -1) if count % d == 0)
        if count // low / low <= 200:
            return count
    return 0


def _encoder_item_tokens(metadata: dict[str, Any], seq_len: int) -> dict[str, int]:
    """Merged tokens per dummy item, by modality, for a traced vision tower
    (``_encoder_batch``); empty when the fields that size it are missing."""
    family = metadata.get("vision_family")
    merge = int(metadata.get("vision_merge_size") or 0)
    patch = int(metadata.get("vision_patch_size") or 0)
    temporal = int(metadata.get("vision_temporal_patch_size") or 0)
    if not (merge and patch and temporal):
        return {}
    per_item: dict[str, int] = {}
    if family == "qwen3_vl":
        unit = patch * merge
        if metadata.get("image_longest_edge"):
            per_item["image"] = _qwen_vl_merged_tokens(int(metadata["image_longest_edge"]), unit)
        if metadata.get("video_longest_edge"):
            per_item["video"] = _qwen_vl_merged_tokens(
                int(metadata["video_longest_edge"]) // temporal, unit
            )
    elif family == "muse_glimmer":
        if metadata.get("max_image_tokens"):
            per_item["image"] = int(metadata["max_image_tokens"])
        frame_tokens = int(metadata.get("max_video_frame_tokens") or 0)
        if frame_tokens and metadata.get("video_num_frames"):
            groups = min(
                max(1, seq_len // frame_tokens),
                max(1, int(metadata["video_num_frames"]) // temporal),
            )
            per_item["video"] = groups * frame_tokens
    return {name: count for name, count in per_item.items() if count > 0}


def _encoder_batch(
    metadata: dict[str, Any], inputs: CalculatorInput, tokens: int
) -> tuple[int, int]:
    """(items, patches per item) of vLLM's dummy encoder batch (traced, v0.30.0).

    profile_run encodes max_items copies of the modality with the most tokens
    per item (encoder_budget.py:203-220; ties go to the later name): items =
    max(1, min(budget // per_item, max_num_seqs x max(1, min(limit,
    max_model_len // per_item)))), budget = max(max_num_batched_tokens,
    per_item) (encoder_budget.py:147-185; encoder_cache_manager.py:333-338;
    chunked prefill on).  Tokens per item (``_encoder_item_tokens``):
    - Qwen3-VL: image = the image with most features under the processor's
      size.longest_edge; video = two frames under video longest_edge /
      temporal_patch_size (qwen2_vl.py:872-879, 969-1021; qwen3_vl.py:
      1013-1036).
    - Muse-Glimmer: image = max_image_tokens; video = min(max_model_len //
      frame_tokens, num_frames // temporal) x frame_tokens (muse_glimmer.py:
      182-222).
    Patches = tokens x merge^2.  (0, 0) when a field is missing.
    """
    seq_len = int(
        inputs.execution_spec_data.get("max_model_len")
        or (
            inputs.workload.input_length + inputs.workload.output_length  # type: ignore[union-attr]
            if inputs.workload.kind == "text"
            else 2048
        )
    )
    per_item = _encoder_item_tokens(metadata, seq_len)
    if not per_item:
        return 0, 0
    _, largest = max((count, name) for name, count in per_item.items())
    item_tokens = per_item[largest]
    merge = int(metadata["vision_merge_size"])
    budget = max(tokens, item_tokens)
    seqs = int(inputs.execution_spec_data.get("max_num_seqs", 256))
    per_prompt = max(1, seq_len // item_tokens)  # no per-modality limit set
    items = max(1, min(budget // item_tokens, seqs * per_prompt))
    return items, item_tokens * merge * merge


def encoder_peak_note(config: dict[str, Any], processor: dict[str, Any] | None) -> str | None:
    """Why the encoder moment of the startup peak (``_encoder_peak_bytes``) is
    not in a multimodal wrapper's prediction, or None when it is (or the
    checkpoint declares no tower).

    The moment counts only a traced vision tower (``_VISION_FAMILIES``) whose
    processor files give the dummy item's size; anything else would enter the
    estimate as 0, so the plan says so instead.  An audio tower is never
    traced.  *config* is config.json, *processor* what ``activation_config``
    takes."""
    model_type = str(config.get("model_type") or "unknown")
    gaps: list[str] = []
    if config.get("vision_config"):
        if model_type not in _VISION_FAMILIES:
            gaps.append(f"{model_type} vision tower (not traced)")
        elif processor is None:
            gaps.append(
                f"{model_type} vision tower (no processor_config.json, "
                "preprocessor_config.json or video_preprocessor_config.json at the revision)"
            )
        else:
            fields = activation_config(config, processor)
            if not fields.get("vision_hidden_size") or not _encoder_item_tokens(fields, 1):
                gaps.append(
                    f"{model_type} vision tower (its config and processor files do not give "
                    "the dummy item's size)"
                )
    if config.get("audio_config"):
        gaps.append(f"{model_type} audio tower (not traced)")
    if not gaps:
        return None
    return (
        f"encoder startup peak not modelled for the {' and the '.join(gaps)}; "
        "activation may be under-predicted"
    )


def declares_towers(config: dict[str, Any]) -> bool:
    """Whether config.json declares a vision or audio tower: how the Hub
    checkpoints of vLLM's multimodal wrappers do (``_multimodal_wrapper``)."""
    return any(config.get(k) for k in _TOWER_FIELDS)


def _encoder_peak_bytes(
    metadata: dict[str, Any], inputs: CalculatorInput, tp: int, tokens: int, dtype_bytes: int
) -> int:
    """The vision tower's peak while vLLM profiles the dummy encoder batch
    (encoder_runner.py:139; peak probe 2026-09-28, vLLM v0.30.0, H100).

    Both probed towers peak in the first block's attention.  U = one
    [patches, vision_hidden] activation in the model dtype.  Named tensors
    (traced to the allocation sites the probe shows); how many are live at
    once is observed (peak probe 2026-09-28):
    - the block input chain, 2U: Qwen3-VL's position embeddings
      (qwen3_vl.py:751) and patch embeddings + them (qwen3_vl.py:867);
      Muse-Glimmer's per-image states (muse_glimmer.py:1014) and their
      concatenation (muse_glimmer.py:1025);
    - the pre-attention norm, U (qwen3_vl.py:491; muse_glimmer.py:710);
    - the qkv projection, 3U (qwen2_5_vl.py:413; muse_glimmer.py:640);
    - q and k stacked contiguous for the rotary, 2U (qwen2_5_vl.py:429;
      muse_glimmer.py:649);
    - then the larger of two moments.  Rotary: its output, 2U (flash-attn
      triton rotary.py:188) -- in FP32 for Muse-Glimmer, whose ApplyRotaryEmb
      computes in FP32 (enable_fp32_compute, muse_glimmer.py:622-626): the FP32
      input copy 4U (rotary_embedding/common.py:188), the FP32 output 4U and
      the cast back 2U (common.py:203).  Attention: the rotary output 2U, the
      ViT flash-attention output U (vit_attn_wrappers.py:62) and the output
      projection U (qwen2_5_vl.py:459).
    - the pixels copied to the GPU (multimodal/inputs.py:326): Qwen3-VL's
      flattened patches, patches x 3 x temporal x patch^2, arrive in bf16;
      Muse-Glimmer's images, patches x 3 x patch^2, in FP32 plus a bf16 copy
      of the last image (muse_glimmer.py:982) -- the dtypes are observed.
    Not modelled (under 3% of either peak): the rotary cos/sin tables
    (qwen3_vl.py:729-730; muse_glimmer.py:1015-1027), a 32 MiB buffer in the
    patch embedding (conv.py:174; muse_glimmer.py:998).  Head-sharded tensors
    are divided by the tensor-parallel size -- inferred, no multi-GPU probe.
    0 for any other tower (not traced; Gemma 4's encoder is chunked by free
    memory, gemma4_mm.py:1254-1282, and its probe peaked in the forward) or
    without the processor's limits; the plan then says so
    (``encoder_peak_note``).
    """
    family = metadata.get("vision_family")
    hidden = int(metadata.get("vision_hidden_size") or 0)
    items, per_item = _encoder_batch(metadata, inputs, tokens)
    patches = items * per_item
    if family not in ("qwen3_vl", "muse_glimmer") or not (hidden and patches):
        return 0
    unit = patches * hidden * dtype_bytes
    sharded = -(-unit // tp)
    base = 2 * unit + unit + 3 * sharded + 2 * sharded
    attention = 2 * sharded + sharded + unit
    patch = int(metadata["vision_patch_size"])
    if family == "muse_glimmer":
        fp32_qk = -(-(2 * patches * hidden * 4) // tp)  # stacked q, k in FP32
        rotary = fp32_qk + fp32_qk + 2 * sharded  # FP32 copy, FP32 output, cast back
        pixels = patches * 3 * patch * patch * 4 + per_item * 3 * patch * patch * 2
    else:
        rotary = 2 * sharded
        temporal = int(metadata["vision_temporal_patch_size"])
        pixels = patches * 3 * temporal * patch * patch * 2
    return base + max(rotary, attention) + pixels


def _multimodal_wrapper(metadata: dict[str, Any], inputs: CalculatorInput) -> bool:
    """A checkpoint with a vision or audio tower, served with its towers on.

    vLLM then profiles the encoder first and feeds the language model
    ``inputs_embeds`` from a preallocated buffer, so the embedding lookup is
    not part of the compiled graph (v1/worker/gpu/model_runner.py:1672-1699,
    model_states/default.py:91-95).  ``--language-model-only`` sets every
    modality limit to 0 and the model is compiled as text-only (peak probe
    2026-09-28: Gemma 4 31B, 2,995,739,688 B logged = the embedding rule).
    Inferred from the config: a ``vision_config`` / ``audio_config`` is how
    the Hub checkpoints of vLLM's SupportsMultiModal wrappers declare towers;
    Apron's engine facts do not list which architectures are multimodal."""
    if inputs.execution_spec_data.get("language_model_only"):
        return False
    return any(metadata.get(k) for k in _TOWER_FIELDS)


def _mamba2_piece_bytes(metadata: dict[str, Any], tp: int, tokens: int, dtype_bytes: int) -> int:
    """A Mamba-2 hybrid's live activations at the end of the compiled piece
    between two Mamba-2 mixers (vLLM v0.30.0), without the fused-MoE workspace.

    The mixer is a splitting op (config/compilation.py:775), so each compiled
    piece runs from one mixer to the next.  Its inputs stay live for the whole
    piece -- observed: in the Gemma 4 probe (2026-09-28) the attention output
    and the residual allocated by the previous piece were live at the MLP peak
    of the next one; the split graph's caller holds them until the piece
    returns.  Named tensors, all [tokens, width] in the model dtype:
    - held from the previous mixer: its in_proj output, width 2 x d_inner +
      2 x groups x state + heads (mamba_mixer2.py:348-360, 573), its SSM output,
      width d_inner (mamba_mixer2.py:578-585; the gate is a view of the
      in_proj output, :600), and the residual, width H;
    - made by the piece for the next mixer: the new residual (H), then its
      in_proj output, then its SSM output (d_inner) -- the normed input (H)
      is freed after the projection, so the larger of the two counts.
    d_inner = mamba_num_heads x mamba_head_dim (models/nemotron_h.py:381);
    heads, groups and d_inner are split by the tensor-parallel size.  The
    profile run passes no attention metadata (v1/worker/gpu/model_runner.py:
    955-957, 1748-1770), so the mixer itself only copies its conv input once
    (mamba_mixer2.py:765-773), below this moment.  Not modelled: the MLP
    moment inside the piece (NemotronH-3.5's shared expert, [tokens, 3712]
    twice, stays under the piece's end).  0 without the Mamba-2 fields.
    """
    heads = int(metadata.get("mamba_num_heads") or 0)
    head_dim = int(metadata.get("mamba_head_dim") or 0)
    groups = int(metadata.get("n_groups") or 0)
    state = int(metadata.get("ssm_state_size") or 0)
    if not (heads and head_dim and groups and state):
        return 0
    heads_tp = -(-heads // tp)
    inner = heads_tp * head_dim
    in_proj = 2 * inner + 2 * -(-groups // tp) * state + heads_tp
    t_h = tokens * int(metadata["hidden_size"]) * dtype_bytes
    projected = tokens * in_proj * dtype_bytes
    ssm_out = tokens * inner * dtype_bytes
    held = projected + ssm_out + t_h
    return held + t_h + projected + max(t_h, ssm_out)


def fp8_block_quant(config: dict[str, Any]) -> bool:
    """A block-FP8 checkpoint: ``quant_method`` fp8 with 128 x 128 weight
    blocks and dynamic activations.  Its MoE layers take Fp8MoEMethod with the
    128-block weight key and the dynamic per-128 activation key
    (layers/quantization/fp8.py:509-541, vLLM v0.30.0).  *config* is
    config.json (``text_config`` read too) or the calculator's merged metadata."""
    text = config.get("text_config")
    nested = text if isinstance(text, dict) else {}
    quant = config.get("quantization_config") or nested.get("quantization_config") or {}
    return (
        isinstance(quant, dict)
        and quant.get("quant_method") == "fp8"
        and list(quant.get("weight_block_size") or []) == [128, 128]
        and quant.get("activation_scheme", "dynamic") == "dynamic"
    )


def _trtllm_fp8_experts(metadata: dict[str, Any], inputs: CalculatorInput, tp: int) -> bool:
    """Whether vLLM v0.30.0 runs the MoE on FlashInfer's monolithic TRTLLM FP8
    kernel, which takes the router logits and allocates its own scratch: no
    fused-MoE workspace (modular_kernel.py:1549-1612 calls the experts without
    ``_allocate_buffers``, the only user of the workspace manager there,
    1117-1196; trtllm_fp8_moe.py:425-509).

    - ``--moe-backend`` auto (config/kernel.py:248) tries FLASHINFER_TRTLLM
      first on CUDA once AITER is out, TRITON only moved ahead on SM 9.0
      (oracle/fp8.py:81-123, 423-441), monolithic before modular
      (oracle/fp8.py:140-146); ``flashinfer_trtllm`` asks for it (366-390);
    - the kernel runs on the SM 10.x family with FlashInfer's TRTLLM fused MoE
      (trtllm_fp8_moe.py:161-169; assumed installed, as for the 16-bit TRTLLM
      experts the 2x B200 Qwen3.8 log showed);
    - block FP8 (``fp8_block_quant``) whose per-rank intermediate size is a
      multiple of 128, else the scales are refined and only Triton takes the
      refined key (oracle/fp8.py:272-302; fp8.py:509-541);
    - routing: renormalized top-k or grouped top-k (trtllm_fp8_moe.py:387-409;
      without ``norm_topk_prob`` the router is Default, refused); at most 512
      experts, a multiple of 4, top-k at most 32 (453-470); a SwiGLU clamp is
      accepted with block FP8 and SiLU (84-102).
    """
    sm = _compute_capability(inputs)
    if sm is None or sm // 10 != 10:
        return False
    backend = str(inputs.execution_spec_data.get("moe_backend") or "auto")
    if backend not in ("auto", "flashinfer_trtllm"):
        return False
    if not (metadata.get("fp8_block_experts") or fp8_block_quant(metadata)):
        return False
    if metadata.get("norm_topk_prob") is False:
        return False
    experts = int(
        metadata.get("n_routed_experts")
        or metadata.get("num_experts")
        or metadata.get("num_local_experts")
        or 0
    )
    top_k = int(metadata.get("num_experts_per_tok") or metadata.get("experts_per_token") or 0)
    width = int(metadata.get("moe_intermediate_size") or 0) // tp
    return 0 < experts <= 512 and experts % 4 == 0 and 0 < top_k <= 32 and width % 128 == 0


# GLM5Next's kpool indexer, vLLM v0.30.0: the prefill K-gather workspace holds
# 40 x max_model_len rows (v1/attention/backends/mla/indexer.py:721-731), each
# ``index_head_dim`` fp8 bytes plus a 4-byte scale, next to a 1 MiB radix top-k
# workspace (layers/sparse_attn_indexer_kpool.py:44, 218-236, 295-304); the
# workspace manager aligns each part to 256 bytes (v1/worker/workspace.py:116-137).
_INDEXER_ROWS_PER_TOKEN = 40
_RADIX_TOPK_WORKSPACE_BYTES = 1 << 20
_WORKSPACE_ALIGN = 256


def _aligned(size: int, alignment: int) -> int:
    return -(-size // alignment) * alignment


def _kpool_indexer_workspace_bytes(metadata: dict[str, Any], seq_len: int) -> int:
    """The workspace the kpool indexer reserves in the profile run (above);
    0 without the indexer's fields."""
    index_dim = int(metadata.get("index_head_dim") or 0)
    if not (index_dim and metadata.get("index_kpool")):
        return 0
    rows = _INDEXER_ROWS_PER_TOKEN * seq_len
    return (
        _aligned(rows * index_dim, _WORKSPACE_ALIGN)
        + _aligned(rows * 4, _WORKSPACE_ALIGN)
        + _aligned(_RADIX_TOPK_WORKSPACE_BYTES, _WORKSPACE_ALIGN)
    )


def _forward_live_bytes(
    metadata: dict[str, Any],
    inputs: CalculatorInput,
    tp: int,
    tokens: int,
    dtype_bytes: int,
    *,
    mla_block: int = _MLA_BLOCK,
) -> int:
    """Bytes live at the peak of the profile run's forward (vLLM v0.29.0).

    Each term is a named tensor with a config shape; *how many* of a shape
    are live at once is a property of the compiled graph, read from the GPU
    memory-history probe (``_dev_notes/cohort-run/peak-probe``) and marked
    "observed".  0 when the config lacks the fields a branch needs.
    """
    hidden = int(metadata["hidden_size"])
    t_h = tokens * hidden * dtype_bytes  # one [tokens, hidden] activation
    heads = int(metadata.get("num_attention_heads") or 0)
    heads_tp = -(-heads // tp) if heads else 0
    experts = (
        metadata.get("n_routed_experts")
        or metadata.get("num_experts")
        or metadata.get("num_local_experts")
    )
    top_k = metadata.get("num_experts_per_tok") or metadata.get("experts_per_token")
    moe_intermediate = metadata.get("moe_intermediate_size") or metadata.get("intermediate_size")
    workspace = 0
    if experts and top_k and moe_intermediate and not _trtllm_fp8_experts(metadata, inputs, tp):
        # (On FlashInfer's monolithic TRTLLM FP8 kernel there is none:
        # ``_trtllm_fp8_experts``.  Its own scratch is not traced -- FlashInfer's
        # source is not in the pinned vLLM tree.)
        # Fused-MoE workspace (traced, workspace.py:207 via modular_kernel.py:
        # 1160-1180; Triton experts' shapes, experts/triton_moe.py:218-233):
        # [tokens, top_k, max(I, H)] shared with the output, plus
        # [tokens, top_k, max(N, H)], in the activation dtype, I the expert
        # width per rank and N = w13's width, 2I gated, I for NemotronH's
        # experts (no gate: models/nemotron_h.py:238; w13 [E, I, H],
        # unquantized_fused_moe_method.py:71-74; modular_kernel.py:810,
        # 875-890).  It persists across layers (workspace manager), so it is
        # live at every later layer's peak.  GLM-4.7-Flash probe: 335,544,320 B,
        # exact; Qwen3.6-35B-A3B-FP8 (v0.30.0) left 536,870,912 B, exact.  Other
        # quantized kernels (MXFP4) may have other shapes -- not probed.
        width = -(-int(moe_intermediate) // tp)
        w13 = width if metadata.get("mlp_hidden_act") else 2 * width
        workspace = tokens * int(top_k) * (max(width, hidden) + max(w13, hidden)) * dtype_bytes

    mamba = _mamba2_piece_bytes(metadata, tp, tokens, dtype_bytes)
    if mamba:
        return mamba + workspace

    kv_lora = metadata.get("kv_lora_rank")
    qk_nope = metadata.get("qk_nope_head_dim")
    v_head = metadata.get("v_head_dim")
    if kv_lora and qk_nope and v_head and heads:
        # MLA.  The profile run has no attention metadata, and each MLA layer
        # allocates the worst-case up-projected prefill context to simulate it
        # (traced, mla_attention.py:779-790): [W, heads, qk_nope + v_head],
        # W = round_up(min(max(8 * max_model_len, 4 * max_num_seqs * block),
        # 64k), block) (mla_attention.py:2158-2183).
        seq_len = int(
            inputs.execution_spec_data.get("max_model_len")
            or (
                inputs.workload.input_length + inputs.workload.output_length  # type: ignore[union-attr]
                if inputs.workload.kind == "text"
                else 2048
            )
        )
        # The block is the KV cache's (v0.30.0: mla_attention.py:903-916,
        # 2286-2311, 1983-1995): 16 unless a hybrid layout aligns it (the
        # caller's *mla_block*; GLM5Next at TP 2: 2176, so W = 67,456).
        seqs = int(inputs.execution_spec_data.get("max_num_seqs", 256))
        width = min(max(8 * seq_len, 4 * seqs * mla_block), _MLA_WORKSPACE_CAP)
        width = -(-max(width, mla_block) // mla_block) * mla_block
        prefill_context = width * heads_tp * (int(qk_nope) + int(v_head)) * dtype_bytes
        # GLM5Next's kpool indexer reserves its prefill workspace before the
        # layer's MLA runs; the workspace manager keeps one buffer, the larger
        # of the indexer's and the fused-MoE request (workspace.py:116-137,
        # 139-209), live at every later layer's peak.
        workspace = max(workspace, _kpool_indexer_workspace_bytes(metadata, seq_len))
        # The compiled graph's buffers live at that moment -- observed (peak
        # probe 2026-09-28, GLM-4.7-Flash, inductor code around
        # mla_attention.py:732): five [tokens, heads x head width] (query /
        # attention output; GLM's qk and v head widths are both 256, so which
        # width each one has is not separable), seven [tokens, hidden] and two
        # [tokens, kv_lora_rank].
        head_width = max(int(qk_nope) + int(metadata.get("qk_rope_head_dim") or 0), int(v_head))
        graph = (
            5 * tokens * heads_tp * head_width * dtype_bytes
            + 7 * t_h
            + 2 * tokens * int(kv_lora) * dtype_bytes
        )
        return prefill_context + workspace + graph

    if not heads:
        return 0
    head_dim = int(metadata.get("head_dim") or hidden // heads)
    head_dim = max(head_dim, int(metadata.get("global_head_dim") or 0))
    # The attention output vLLM allocates before the attention op
    # (attention.py:519): [tokens, heads x head_dim], the widest layer's.
    attention_out = tokens * heads_tp * head_dim * dtype_bytes
    if workspace:
        # MoE with GQA attention: the dense layers' attention buffers plus the
        # workspace in place of the MLP buffers.  Inferred -- no probe of a
        # GQA MoE model.
        return attention_out + 2 * t_h + workspace
    intermediate = metadata.get("intermediate_size")
    if not intermediate:
        return 0
    width = -(-int(intermediate) // tp)
    # Observed (peak probe 2026-09-28, Gemma 4 31B multimodal): the peak falls
    # in the compiled piece after a global-attention layer, with the attention
    # output, two [tokens, hidden] (o_proj output and residual), the MLP's
    # gate_up [tokens, 2I] and its activation [tokens, I] live at once.
    return attention_out + 2 * t_h + tokens * 3 * width * dtype_bytes


def _activation_estimate(
    metadata: dict[str, Any], inputs: CalculatorInput, tp: int, *, mla_block: int = _MLA_BLOCK
) -> int | None:
    """Peak activation vLLM measures in its startup profiling run (cold compile).

    The profile run's torch peak is the largest of four moments; T is
    max_num_batched_tokens, b the model dtype's bytes, V the padded vocabulary.

    1. Compile (TP 1, text-only models).  The first forward compiles the
       model with TorchInductor's combo-kernel benchmarking on
       (config/compilation.py:983-993); the embedding lookup is a Triton
       kernel, and the benchmark allocates a random tensor of the embedding
       plus the kernel's two outputs: V x H x b + 2 x T x H x b (traced, L5
       review; peak probe 2026-09-28: Gemma 4 31B --language-model-only
       2,994,733,056 predicted, 2,995,739,688 logged).  With tensor
       parallelism the embedding is vLLM's own CUDA op
       (vocab_parallel_embedding.py:331-335, 503), not benchmarked; a
       multimodal wrapper takes ``inputs_embeds`` and the embedding is not in
       the graph at all (``_multimodal_wrapper``).  On a warm compile cache
       this transient does not occur; the cohort's boots were cold.
    2. Forward (``_forward_live_bytes``): the compiled graph's live buffers
       plus, for MLA, the per-layer prefill-context dummy and, for MoE, the
       persistent fused-MoE workspace; for a Mamba-2 hybrid (NemotronH), the
       piece between two mixers with its held inputs
       (``_mamba2_piece_bytes``); a multimodal wrapper also still holds
       the dummy encoder outputs (encoder_runner.py:139-147), at most the
       encoder budget, max(T, tokens per item) (encoder_cache_manager.py:
       333-338; encoder_budget.py:160-185) x H x b -- counted as T x H x b.
       Peak probe 2026-09-28: GLM-4.7-Flash 2,181,038,080 predicted,
       2,190,433,320 logged; Gemma 4 31B multimodal 1,589,641,216 predicted,
       1,599,875,317 logged (the probe also shows a 32 MiB buffer left by the
       vision tower, modeling_gemma4.py:619, not modelled).  No fused-MoE
       workspace on FlashInfer's TRTLLM FP8 kernel (``_trtllm_fp8_experts``);
       GLM5Next's kpool indexer workspace shares the workspace buffer.
       GLM-5.3-Flash on 2x B200 (v0.30.0, no probe): 4,833,935,360 predicted
       at the boot's shape (the MLA dummy at W = 67,456, the indexer's
       174,063,616 B), 4,702,989,189 logged; its compiled-graph buffer count
       is GLM-4.7-Flash's, not probed for GLM5Next's mHC / KDA graph.
    3. Sampler: the logits for min(T, max_num_seqs) rows -- a second copy
       while a final soft-cap is applied (logits_processor.py:113-116) -- and
       the hidden states; with tensor parallelism the local shard, the
       all-gathered logits and their contiguous copy (logits_processor.py:130;
       base_device_communicator.py:239-251).  The dummy sampler run adds no
       FP32 copy (no request needs logits processing, v1/worker/gpu/sample/
       sampler.py:218-222).
    4. Encoder (multimodal wrappers, ``_encoder_peak_bytes``): before the text
       forward, profile_run encodes a dummy batch of the largest multimodal
       item (encoder_runner.py:111-147); the vision tower's first block holds
       [patches, vision hidden] activations.  Peak probe 2026-09-28 (vLLM
       v0.30.0): Qwen3.6-35B-A3B-FP8 2,013,265,920 predicted, 2,061,584,302
       logged; Muse-Glimmer-30B 1,908,277,248 predicted, 1,964,947,537 logged.
       Needs the processor's limits (``activation_config(config, processor)``)
       and a traced tower; otherwise 0.

    ``None`` when the config lacks the vocabulary or hidden size.
    """
    vocab = metadata.get("vocab_size")
    hidden = metadata.get("hidden_size")
    if not vocab or not hidden:
        return None
    dtype_bytes = DTYPE_BYTES.get(metadata.get("torch_dtype", "bfloat16"), 2)
    tokens = int(inputs.execution_spec_data.get("max_num_batched_tokens", 2048))
    padded = -(-int(vocab) // VOCAB_PADDING) * VOCAB_PADDING
    t_h = tokens * int(hidden) * dtype_bytes
    multimodal = _multimodal_wrapper(metadata, inputs)

    compile_peak = (
        padded * int(hidden) * dtype_bytes + 2 * t_h if tp == 1 and not multimodal else 0
    )

    forward = _forward_live_bytes(metadata, inputs, tp, tokens, dtype_bytes, mla_block=mla_block)
    if multimodal:
        forward += t_h  # the dummy encoder outputs, bounded by the budget

    rows = min(tokens, int(inputs.execution_spec_data.get("max_num_seqs", 256)))
    logits = rows * padded * dtype_bytes
    if tp == 1:
        sampler = logits * (2 if metadata.get("final_logit_softcapping") else 1) + t_h
    else:
        sampler = 2 * logits + -(-logits // tp) + t_h
    encoder = _encoder_peak_bytes(metadata, inputs, tp, tokens, dtype_bytes) if multimodal else 0
    return max(compile_peak, forward, sampler, encoder)


def _tensor_parallel(inputs: CalculatorInput) -> int:
    return max(1, int(inputs.execution_spec_data.get("tensor_parallel", 1) or 1))


def _compute_capability(inputs: CalculatorInput) -> int | None:
    """The GPU's compute capability as major * 10 + minor ("9.0" -> 90);
    None when unknown ("0.0") or unreadable."""
    major, _, minor = str(inputs.hardware.compute_capability).partition(".")
    if not major.isdigit() or not (minor or "0").isdigit() or major == "0":
        return None
    return int(major) * 10 + int(minor or "0")


def _per_gpu(total: int, tp: int) -> int:
    """A tensor-parallel shard of *total* bytes (vLLM splits the linear and
    embedding weights across ranks; the replicated norms are negligible)."""
    return -(-total // tp)


def _extract_gqa_params(metadata: dict[str, Any]) -> dict[str, int] | None:
    """Extract GQA parameters from config.json-style metadata."""
    num_layers = metadata.get("num_hidden_layers")
    num_kv_heads = metadata.get("num_key_value_heads")
    hidden_size = metadata.get("hidden_size")
    num_attention_heads = metadata.get("num_attention_heads")

    if num_layers is None or num_kv_heads is None:
        return None
    if hidden_size is None or num_attention_heads is None:
        return None
    if num_attention_heads == 0:
        return None

    head_dim: int = metadata.get("head_dim", hidden_size // num_attention_heads)

    return {
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": head_dim,
        "num_attention_heads": int(num_attention_heads),
    }


def _apply_sliding_window(
    kv_per_token: int,
    seq_len: int,
    max_batch_size: int,
    metadata: dict[str, Any],
) -> tuple[int, int]:
    """Cap effective sequence length at sliding_window when present.

    Returns (kv_cache_bytes, effective_seq_len).
    """
    sliding_window = metadata.get("sliding_window")
    if sliding_window is not None and sliding_window > 0:
        effective = min(seq_len, sliding_window)
    else:
        effective = seq_len
    return kv_per_token * effective * max_batch_size, effective


# ---------------------------------------------------------------------------
# The KV budget: what vLLM leaves for the KV cache after its startup profile
# ---------------------------------------------------------------------------
#
# vLLM v0.29.0 and v0.30.0 alike (v1/worker/gpu_worker.py:606-610 / 621-625;
# utils/mem_utils.py:317-326; v1/worker/utils.py:521-523 / 539-541):
#
#   available = ceil(total x gpu_memory_utilization)
#               - (total_consumed + transient_peak_headroom)
#               - cudagraph_memory_estimate
#
# ``total_consumed`` is the free memory lost since the snapshot vLLM takes after
# the CUDA context and NCCL are up (gpu_worker.py:414-440 / 432-451), so neither is
# subtracted from the request.  In the terms vLLM logs, the subtracted memory is
# the weights + the torch peak increase (``_activation_estimate``) + the rest
# ("non-torch" below), and the CUDA-graph estimate is subtracted too
# (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS defaults to 1, envs.py:2105 / 2109).
# Every cohort record satisfies this identity to the log's 0.01 GiB; the terms
# per record are in ``_dev_notes/cohort-run/kv-budget-residuals.md``.

# Measured per GPU class over every healthy cohort record (medians over the 25
# distinct boots; ranges in kv-budget-residuals.md).  "sm90": the H100 boots;
# "sm8x": RTX 4090, L4, RTX A6000, A100.  The two classes also differ in vLLM's
# batch tier (8192 tokens / 1024 sequences vs 2048 / 256), so the records
# cannot tell GPU architecture from tier apart.  An unknown GPU takes the larger.
#
# Non-torch base: what is left of the logged non-KV memory after the weights,
# the torch peak, the sampler state, the encoder's embeddings buffer and a
# retained compile segment: 0.05-0.16 GiB on sm8x, 0.21-0.39 GiB on sm90.
_NON_TORCH_BASE_BYTES = {"sm8x": 112 << 20, "sm90": 307 << 20}
# CUDA-graph estimate per decoder layer and FULL decode graph: 0.20-0.63 MiB
# on sm8x (dense attention 0.33-0.39), 0.25-1.39 MiB on sm90.
_CUDA_GRAPH_BYTES_PER_LAYER_GRAPH = {"sm8x": 374 << 10, "sm90": 537 << 10}


def _gpu_class(inputs: CalculatorInput) -> str:
    sm = _compute_capability(inputs)
    return "sm8x" if sm is not None and sm < 90 else "sm90"


def requested_memory_bytes(inputs: CalculatorInput) -> int:
    """What vLLM requests: ceil(total x gpu_memory_utilization)
    (v1/worker/utils.py:521-523 in v0.29.0, 539-541 in v0.30.0).  The
    utilization is the plan's; 0.90 is what the planner's plans set
    (plan_builder.PLAN_GPU_MEMORY_UTILIZATION)."""
    utilization = float(inputs.execution_spec_data.get("gpu_memory_utilization", 0.90))
    return math.ceil(inputs.hardware.total_memory_bytes * utilization)


def sampler_state_bytes(vocab_size: int, max_num_seqs: int) -> int:
    """GPU tensors the model runner's sampler allocates after the weights, one
    row per request slot (vLLM v0.29.0 and v0.30.0 model runner V2, the
    default runner): the penalties' output-token counts, [max_num_seqs, V]
    int32 (v1/worker/gpu/sample/penalties.py:40-42), and prompt bitmask,
    [max_num_seqs, ceil(V / 32)] int32 (penalties.py:32-37), and the
    structured-output bitmask, [max_num_seqs, ceil(V / 32)] int32 without
    speculative decoding (v1/worker/gpu/structured_outputs.py:51-53 / 50-52).
    V is the unpadded vocabulary (``get_vocab_size``).  Peak probe 2026-09-28:
    Gemma 4 31B 1,140,850,688 B of these among 1,247,013,131 B allocated
    between the weights and the profile run; GLM-4.7-Flash 674,037,760 among
    696,961,414."""
    words = -(-int(vocab_size) // 32)
    return int(max_num_seqs) * (int(vocab_size) + 2 * words) * 4


def non_torch_bytes(metadata: dict[str, Any], inputs: CalculatorInput) -> int:
    """The non-KV memory beyond the weights and the torch peak: the sampler
    state (``sampler_state_bytes``), a multimodal wrapper's encoder embeddings
    buffer, [max_num_batched_tokens, H] in the model dtype
    (v1/worker/gpu/mm/encoder_runner.py:58-60), and the measured base of the
    GPU class (``_NON_TORCH_BASE_BYTES``: CUDA and library memory allocated
    after vLLM's snapshot, allocator slack)."""
    seqs = int(inputs.execution_spec_data.get("max_num_seqs", 256))
    tokens = int(inputs.execution_spec_data.get("max_num_batched_tokens", 2048))
    vocab = int(metadata.get("vocab_size") or 0)
    hidden = int(metadata.get("hidden_size") or 0)
    dtype_bytes = DTYPE_BYTES.get(metadata.get("torch_dtype", "bfloat16"), 2)
    embeds = tokens * hidden * dtype_bytes if _multimodal_wrapper(metadata, inputs) else 0
    return sampler_state_bytes(vocab, seqs) + embeds + _NON_TORCH_BASE_BYTES[_gpu_class(inputs)]


def compile_segment_bytes(metadata: dict[str, Any], inputs: CalculatorInput, tp: int) -> int:
    """The compile benchmark's embedding-sized tensor (moment 1 of
    ``_activation_estimate``), when vLLM may still hold its segment.

    In 10 of the 21 distinct cohort boots where that moment applies (TP 1,
    text-only) the logged non-KV memory holds one more V x H x b block (0.90-
    1.17 of it, over the class base) that no live torch tensor accounts for
    after the profile run; in 10 it holds none (under 0.04 of it); Qwen3-0.6B-
    FP8 on an H100 holds 0.76 of one.  The same checkpoint went both ways on two GPUs
    (Qwen3-32B: A100 none, H100 1.45 GiB), so it is the caching allocator's
    placement, not the model: the prediction leaves it out and the plan counts
    it as uncertainty (the state-block guard subtracts it).  0 where the
    moment does not apply."""
    vocab, hidden = metadata.get("vocab_size"), metadata.get("hidden_size")
    if tp != 1 or not vocab or not hidden or _multimodal_wrapper(metadata, inputs):
        return 0
    dtype_bytes = DTYPE_BYTES.get(metadata.get("torch_dtype", "bfloat16"), 2)
    return -(-int(vocab) // VOCAB_PADDING) * VOCAB_PADDING * int(hidden) * dtype_bytes


def _capture_sizes(max_num_seqs: int, inputs: CalculatorInput) -> list[int]:
    """vLLM's default CUDA-graph capture sizes: 1, 2, 4, then steps of 8 to 248
    and of 16 from 256 to the ceiling min(2 x max_num_seqs, 512), 1024 on
    SM100 (config/vllm.py:2006-2014, 2119-2132 in v0.29.0; 2217-2225,
    2330-2343 in v0.30.0)."""
    sm = _compute_capability(inputs)
    ceiling = min(2 * max_num_seqs, 1024 if sm is not None and sm // 10 == 10 else 512)
    sizes = [s for s in (1, 2, 4) if s <= ceiling]
    sizes += list(range(8, min(ceiling + 1, 256), 8))
    sizes += list(range(256, ceiling + 1, 16)) if ceiling >= 256 else []
    return sizes


def cuda_graph_estimate_bytes(metadata: dict[str, Any], inputs: CalculatorInput) -> int:
    """The CUDA-graph memory vLLM estimates and subtracts from the KV budget.

    vLLM captures every graph once into a throwaway pool, FULL decode graphs
    only for the two largest sizes, and extrapolates the rest as (count - 1)
    x the second one's cost (v1/worker/gpu/cudagraph_utils.py:718-831 in
    v0.29.0, 847-960 in v0.30.0).  Inferred: a graph's cost grows with its
    kernels, so the estimate is layers x FULL decode graphs (capture sizes up
    to max_num_seqs, cudagraph_utils.py:250-274 in v0.29.0) x the class's
    measured cost (``_CUDA_GRAPH_BYTES_PER_LAYER_GRAPH``; the dense-attention
    records on sm8x agree to +-8%).  The estimate is 1.0-2.6 x the memory the
    real capture then takes (the logged "actual"); vLLM subtracts the
    estimate.  Not modelled: MLA (GLM-4.7-Flash 1.39 MiB per layer-graph on
    sm90, DeepSeek-V2-Lite 0.63 on A100 -- one point per class) and Mamba-1
    (0.20).  0 with ``enforce_eager``.

    The estimate also holds what the model allocates and keeps on its first
    forward with attention metadata, the warmup before the first graph
    (``layered.first_capture_bytes``, added by the caller): GLM-5.3-Flash on
    2x B200 logged 0.09 GiB actual, 1.68 GiB estimated (this rule: 0.09)."""
    if inputs.execution_spec_data.get("enforce_eager", False):
        return 0
    layers = int(metadata.get("num_hidden_layers") or 0)
    seqs = int(inputs.execution_spec_data.get("max_num_seqs", 256))
    graphs = sum(1 for size in _capture_sizes(seqs, inputs) if size <= seqs)
    return layers * graphs * _CUDA_GRAPH_BYTES_PER_LAYER_GRAPH[_gpu_class(inputs)]


def _common_overhead(
    weight_bytes: int,
    kv_cache: int,
    activation_estimate: int,
    inputs: CalculatorInput,
    metadata: dict[str, Any],
    *,
    first_capture_bytes: int = 0,
) -> dict[str, int]:
    """The KV budget and its terms, shared across mechanism branches.

    *first_capture_bytes*: what the model allocates and keeps on its first
    forward with attention metadata, which falls inside vLLM's CUDA-graph
    memory estimate (``layered.first_capture_bytes``); none with
    ``enforce_eager``, where vLLM estimates nothing."""
    non_pytorch_overhead = non_torch_bytes(metadata, inputs)
    cuda_graph_estimate = cuda_graph_estimate_bytes(metadata, inputs)
    if not inputs.execution_spec_data.get("enforce_eager", False):
        cuda_graph_estimate += first_capture_bytes

    total_required = (
        weight_bytes + kv_cache + activation_estimate + non_pytorch_overhead + cuda_graph_estimate
    )

    gpu_available = requested_memory_bytes(inputs)
    available_kv_cache = (
        gpu_available
        - weight_bytes
        - activation_estimate
        - non_pytorch_overhead
        - cuda_graph_estimate
    )

    return {
        "non_pytorch_overhead_bytes": non_pytorch_overhead,
        "cuda_graph_estimate_bytes": cuda_graph_estimate,
        "total_required_bytes": total_required,
        "available_kv_cache_bytes": available_kv_cache,
        "gpu_available_bytes": gpu_available,
        "compile_segment_bytes": compile_segment_bytes(metadata, inputs, _tensor_parallel(inputs)),
    }


def _workload_params(inputs: CalculatorInput) -> tuple[int, int, int]:
    """Extract ISL, OSL, max_batch_size from inputs."""
    isl, osl = 512, 128
    if inputs.workload.kind == "text":
        isl = inputs.workload.input_length  # type: ignore[union-attr]
        osl = inputs.workload.output_length  # type: ignore[union-attr]
    max_batch_size = inputs.execution_spec_data.get("max_batch_size", 4)
    return isl, osl, max_batch_size


# ---------------------------------------------------------------------------
# Branch: autoregressive_decode (GQA / MHA / MQA)
# ---------------------------------------------------------------------------


@register_calculator("autoregressive_decode")
def calculate_autoregressive_decode(
    inputs: CalculatorInput,
) -> dict[str, Any] | None:
    """Dense GQA/MHA/MQA memory prediction."""
    metadata = inputs.artifact_metadata
    gqa = _extract_gqa_params(metadata)
    if gqa is None:
        return None

    dtype_bytes = _kv_dtype_bytes(metadata, inputs)
    tp = _tensor_parallel(inputs)

    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None
    from apron.domain.mechanisms.layered import padded_weight_bytes

    # Per GPU: each tensor-parallel rank holds 1/tp of the weights and of the
    # KV heads (a KV head is replicated when there are fewer heads than ranks);
    # vLLM pads some experts past the checkpoint (``padded_weight_bytes``:
    # gpt-oss's MXFP4 experts on Marlin below SM 9.0).
    weight_bytes = _per_gpu(int(weight_bytes), tp) + padded_weight_bytes(
        metadata,
        tp=tp,
        sm=_compute_capability(inputs),
        moe_backend=str(inputs.execution_spec_data.get("moe_backend") or "auto"),
    )
    kv_heads_per_gpu = max(1, -(-gqa["num_kv_heads"] // tp))

    isl, osl, max_batch_size = _workload_params(inputs)
    seq_len = isl + osl

    kv_per_token = 2 * gqa["num_layers"] * kv_heads_per_gpu * gqa["head_dim"] * dtype_bytes
    kv_cache, effective_seq = _apply_sliding_window(
        kv_per_token, seq_len, max_batch_size, metadata
    )

    activation_estimate = _activation_estimate(metadata, inputs, tp)
    if activation_estimate is None:
        return None

    overhead = _common_overhead(weight_bytes, kv_cache, activation_estimate, inputs, metadata)

    return {
        "weight_memory_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache,
        "kv_per_token_bytes": kv_per_token,
        "activation_estimate_bytes": activation_estimate,
        **overhead,
        "num_layers": gqa["num_layers"],
        "num_attention_heads": gqa["num_attention_heads"],
        "num_kv_heads": gqa["num_kv_heads"],
        "head_dim": gqa["head_dim"],
        "dtype_bytes": dtype_bytes,
        "isl": isl,
        "osl": osl,
        "max_batch_size": max_batch_size,
        "effective_seq_len": effective_seq,
        "tensor_parallel": tp,
        "mechanism_detail": "gqa",
    }


# ---------------------------------------------------------------------------
# Branch: mla_decode (Multi-Latent Attention — DeepSeek V2/V3/R1)
# ---------------------------------------------------------------------------


def _extract_mla_params(
    metadata: dict[str, Any],
) -> dict[str, int] | None:
    """Extract MLA-specific parameters. Returns None if not an MLA model."""
    kv_lora_rank = metadata.get("kv_lora_rank")
    qk_rope_head_dim = metadata.get("qk_rope_head_dim")
    num_layers = metadata.get("num_hidden_layers")

    if kv_lora_rank is None or qk_rope_head_dim is None or num_layers is None:
        return None

    return {
        "kv_lora_rank": int(kv_lora_rank),
        "qk_rope_head_dim": int(qk_rope_head_dim),
        "num_layers": int(num_layers),
        "num_attention_heads": int(metadata.get("num_attention_heads", 0)),
        "num_kv_heads": int(metadata.get("num_key_value_heads", 0)),
    }


@register_calculator("mla_decode")
def calculate_mla_decode(
    inputs: CalculatorInput,
) -> dict[str, Any] | None:
    """MLA memory prediction — compressed KV, shared across heads."""
    metadata = inputs.artifact_metadata
    mla = _extract_mla_params(metadata)
    if mla is None:
        return None

    dtype_bytes = _kv_dtype_bytes(metadata, inputs)
    tp = _tensor_parallel(inputs)

    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None
    # Per GPU: weights shard across tensor-parallel ranks; the MLA latent
    # cache is shared by all heads, so every rank keeps all of it.
    weight_bytes = _per_gpu(int(weight_bytes), tp)

    isl, osl, max_batch_size = _workload_params(inputs)
    seq_len = isl + osl

    cache_per_layer = (mla["kv_lora_rank"] + mla["qk_rope_head_dim"]) * dtype_bytes
    kv_per_token = cache_per_layer * mla["num_layers"]
    kv_cache, effective_seq = _apply_sliding_window(
        kv_per_token, seq_len, max_batch_size, metadata
    )

    activation_estimate = _activation_estimate(metadata, inputs, tp)
    if activation_estimate is None:
        return None

    overhead = _common_overhead(weight_bytes, kv_cache, activation_estimate, inputs, metadata)

    return {
        "weight_memory_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache,
        "kv_per_token_bytes": kv_per_token,
        "activation_estimate_bytes": activation_estimate,
        **overhead,
        "num_layers": mla["num_layers"],
        "num_attention_heads": mla["num_attention_heads"],
        "num_kv_heads": mla["num_kv_heads"],
        "kv_lora_rank": mla["kv_lora_rank"],
        "qk_rope_head_dim": mla["qk_rope_head_dim"],
        "cache_per_layer_bytes": cache_per_layer,
        "dtype_bytes": dtype_bytes,
        "isl": isl,
        "osl": osl,
        "max_batch_size": max_batch_size,
        "effective_seq_len": effective_seq,
        "tensor_parallel": tp,
        "mechanism_detail": "mla",
    }


# ---------------------------------------------------------------------------
# Branch: ssm_decode (Mamba-1 — a fixed-size state per sequence, no KV cache)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Branch: layered_decode (per-layer caches: hybrid state + attention, Gemma 4)
# ---------------------------------------------------------------------------


@register_calculator("layered_decode")
def calculate_layered_decode(
    inputs: CalculatorInput,
) -> dict[str, Any] | None:
    """Models whose layers hold different caches (``layered.py``): Qwen3.5
    gated delta net + attention, NemotronH Mamba2 + attention, Gemma 4 sliding +
    global attention, and the vLLM v0.30.0 layouts (DeepSeek V4 / V4.1,
    Qwen4Exp, GLM5Next, GLM-5.x DSA, MiniMax-M3).  One request's reservation
    is counted the way vLLM pages it, at the plan's context length.

    The KV part of the v0.30 layouts reads the GPU's compute capability (the
    DeepSeek record and block, GLM-5.3's backend) and the engine's KV dtype
    (DeepSeek forces fp8_ds_mla, QSA and GLM-5.3 are counted in bf16 only).
    ``replicated_weight_bytes`` (plan_pipeline) are weights every
    tensor-parallel rank holds whole instead of a shard, and
    ``kv_head_weight_bytes`` weights split by KV head, which ranks share when
    there are fewer KV heads than ranks; vLLM also allocates weights the
    checkpoint does not store (``duplicated_weight_bytes``) and pads some
    (``padded_weight_bytes``: MXFP4 experts on every arch, 16-bit experts on
    FlashInfer TRTLLM on SM 10.x).  Its "model weights" figure also holds the
    buffers the model builds at load (``startup_buffer_bytes``: GLM5Next's
    top-k index buffers), and its CUDA-graph estimate what the first forward
    with attention metadata allocates and keeps (``first_capture_bytes``).
    MLA's profile-run prefill dummy is aligned to the layout's block.
    """
    from apron.domain.mechanisms.layered import (
        block_accounting,
        duplicated_weight_bytes,
        first_capture_bytes,
        kv_layout,
        layer_kinds,
        padded_weight_bytes,
        startup_buffer_bytes,
    )

    metadata = inputs.artifact_metadata
    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None
    tp = _tensor_parallel(inputs)
    dtype_bytes = _kv_dtype_bytes(metadata, inputs)
    model_dtype_bytes = _DTYPE_BYTES.get(str(metadata.get("torch_dtype")), 2)
    kinds = layer_kinds(
        metadata,
        tp=tp,
        kv_dtype_bytes=dtype_bytes,
        model_dtype_bytes=model_dtype_bytes,
        sm=_compute_capability(inputs),
        kv_cache_dtype=str(inputs.execution_spec_data.get("kv_cache_dtype", "auto")),
    )
    if kinds is None:
        return None
    isl, osl, max_batch_size = _workload_params(inputs)
    seq_len = isl + osl
    batched = int(inputs.execution_spec_data.get("max_num_batched_tokens") or 2048)
    # Two batches in flight with async scheduling (config/vllm.py:563-580).
    blocks = block_accounting(
        kinds, seq_len, in_flight_tokens=2 * batched, layout=kv_layout(metadata)
    )
    if blocks is None:
        return None
    per_sequence = blocks.bytes_per_sequence
    replicated = min(int(metadata.get("replicated_weight_bytes") or 0), int(weight_bytes))
    # Projections split by KV head: with fewer KV heads than ranks each rank
    # holds one head, tp / heads ranks alike (linear.py:1397-1403).
    by_kv_head = min(
        int(metadata.get("kv_head_weight_bytes") or 0), int(weight_bytes) - replicated
    )
    kv_ranks = max(1, min(tp, int(metadata.get("num_key_value_heads") or tp)))
    weight_bytes = (
        _per_gpu(int(weight_bytes) - replicated - by_kv_head, tp)
        + replicated
        + _per_gpu(by_kv_head, kv_ranks)
        + duplicated_weight_bytes(metadata, tp=tp, dtype_bytes=model_dtype_bytes)
        + padded_weight_bytes(
            metadata,
            tp=tp,
            sm=_compute_capability(inputs),
            moe_backend=str(inputs.execution_spec_data.get("moe_backend") or "auto"),
        )
        + startup_buffer_bytes(metadata, max_num_batched_tokens=batched)
    )
    # The KV cache block (the full-attention layers'), to which vLLM aligns
    # MLA's profile-run prefill dummy (mla_attention.py:1983-1995, 2286-2311).
    mla_block = next((k.block_size for k in kinds if k.kind == "full"), _MLA_BLOCK)
    activation_estimate = _activation_estimate(metadata, inputs, tp, mla_block=mla_block)
    if activation_estimate is None:
        return None
    kv_cache = per_sequence * max_batch_size
    overhead = _common_overhead(
        weight_bytes,
        kv_cache,
        activation_estimate,
        inputs,
        metadata,
        first_capture_bytes=first_capture_bytes(
            metadata,
            tp=tp,
            sm=_compute_capability(inputs),
            kv_cache_dtype=str(inputs.execution_spec_data.get("kv_cache_dtype", "auto")),
        ),
    )
    return {
        "weight_memory_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache,
        "kv_per_token_bytes": sum(k.count * k.bytes_per_token for k in kinds if k.kind == "full"),
        "state_per_sequence_bytes": per_sequence,
        # One KV block as vLLM sizes the pool (``available // bytes_per_block``);
        # a state (Mamba) cache serves one decode sequence per block.
        **(
            {"kv_bytes_per_block": blocks.bytes_per_block}
            if any(k.kind == "state" and k.count > 0 for k in kinds)
            else {}
        ),
        "activation_estimate_bytes": activation_estimate,
        **overhead,
        "num_layers": sum(k.count for k in kinds),
        "num_attention_heads": int(metadata.get("num_attention_heads") or 0),
        "num_kv_heads": int(metadata.get("num_key_value_heads") or 0),
        "dtype_bytes": dtype_bytes,
        "isl": isl,
        "osl": osl,
        "max_batch_size": max_batch_size,
        "effective_seq_len": seq_len,
        "tensor_parallel": tp,
        "mechanism_detail": "+".join(f"{k.kind}x{k.count}" for k in kinds),
    }


@register_calculator("ssm_decode")
def calculate_ssm_decode(
    inputs: CalculatorInput,
) -> dict[str, Any] | None:
    """Mamba-1 memory: the state per sequence does not grow with its length.

    Per layer and sequence vLLM v0.29.0 keeps a conv state of
    ``intermediate/tp x (conv_kernel - 1)`` and an SSM state of
    ``intermediate/tp x state_size`` (model_executor/layers/mamba/
    mamba_utils.py:169-181), both in the model dtype when the cache dtypes are
    "auto" (mamba_utils.py:97-109; config/cache.py:184).  Not yet checked
    against a measured boot.
    """
    metadata = inputs.artifact_metadata
    layers = metadata.get("num_hidden_layers")
    intermediate = metadata.get("intermediate_size")
    state_size = metadata.get("state_size")
    conv_kernel = metadata.get("conv_kernel")
    if not (layers and intermediate and state_size and conv_kernel):
        return None
    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None
    tp = _tensor_parallel(inputs)
    dtype_bytes = _kv_dtype_bytes(metadata, inputs)
    weight_bytes = _per_gpu(int(weight_bytes), tp)
    per_rank = -(-int(intermediate) // tp)
    state_per_sequence = (
        int(layers) * per_rank * ((int(conv_kernel) - 1) + int(state_size)) * dtype_bytes
    )
    isl, osl, max_batch_size = _workload_params(inputs)
    state_cache = state_per_sequence * max_batch_size
    activation_estimate = _activation_estimate(metadata, inputs, tp)
    if activation_estimate is None:
        return None
    overhead = _common_overhead(weight_bytes, state_cache, activation_estimate, inputs, metadata)
    return {
        "weight_memory_bytes": weight_bytes,
        "kv_cache_bytes": state_cache,
        "kv_per_token_bytes": 0,
        "state_per_sequence_bytes": state_per_sequence,
        "activation_estimate_bytes": activation_estimate,
        **overhead,
        "num_layers": int(layers),
        "num_attention_heads": 0,
        "num_kv_heads": 0,
        "dtype_bytes": dtype_bytes,
        "isl": isl,
        "osl": osl,
        "max_batch_size": max_batch_size,
        "effective_seq_len": isl + osl,
        "tensor_parallel": tp,
        "mechanism_detail": "mamba1",
    }
