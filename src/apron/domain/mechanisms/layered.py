"""Per-layer cache specs and vLLM v0.29.0's KV page accounting, for models whose
layers are not all alike: hybrid attention + recurrent state (Qwen3.5 gated
delta net, NemotronH Mamba2) and mixed attention (Gemma 4 sliding + global).

Traced from the pinned source (``_dev_notes/cohort-run/hybrid-memory-trace.md``
and ``gemma4-memory-trace.md``; paths under ``.sources/vllm/vllm/``):

- one page size for every layer: a state layer's page sets the attention
  block size (``platforms/interface.py:883-938``); attention layers with
  different per-token bytes are rescaled to the largest page
  (``v1/core/kv_cache_utils.py:1180-1184``);
- layers of one kind are grouped, ``group_size = min`` of the kinds' counts
  (``max`` when ``max < 1.5 * min``), each kind in ``ceil(n / size)`` groups
  (``kv_cache_utils.py:1310-1345``); a block is ``group_size * page`` bytes;
- one request at ``L`` tokens takes, per group: ``ceil(L / block)`` blocks
  for full attention (``kv_cache_interface.py:464-469``);
  ``ceil(min(window - 1 + F, L) / block) + 1`` for sliding attention, ``F``
  the in-flight tokens (``kv_cache_interface.py:723-740``); 2 for a state
  group with prefix caching on, the default for hybrids (1 without)
  (``kv_cache_interface.py:883-895``).

vLLM v0.30.0 pages four more layouts its own way (``kv_layout``; traced in
``_dev_notes/cohort-run/deepseek-v4-kv-trace.md`` and
``qwen4exp-glm5next-kv-trace.md``, paths under ``.sources/vllm-v0.30.0/vllm/``):
DeepSeek V4 / V4.1 and Qwen4Exp through the block-outermost *packed* grouping
(``v1/core/kv_cache_utils.py:1978-2112``), GLM5Next through its own grouping
(``kv_cache_utils.py:1195-1275``).  Their layer kinds carry the page vLLM
builds (``block_size``, ``page_bytes``), alignment padding included.

Pure arithmetic on config values; no I/O.  Predictions until a boot checks them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from typing import Any, NamedTuple

BASE_BLOCK = 16  # config/cache.py:79 and the 16-token kernel alignment
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "fp8": 1, "float8": 1}


@dataclass(frozen=True)
class LayerKind:
    """One kind of cache-holding layer and how many there are."""

    kind: str  # "full", "sliding", "state"; v0.30 layouts add "compressed", "ring", "tail"
    count: int
    bytes_per_token: int = 0  # attention: K and V per context token, per GPU
    window: int = 0  # sliding attention
    state_bytes: int = 0  # recurrent state per sequence, per GPU
    # v0.30 layouts: the page vLLM builds for one layer of this kind, per GPU.
    block_size: int = 0  # tokens per page (the manager block)
    page_bytes: int = 0  # one layer's page, alignment and state padding included
    # State kinds vLLM never buckets together (MambaSpec.tp_replicated,
    # kv_cache_interface.py:1059-1070).
    bucket: str = ""


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """The language model's config (vLLM's ``hf_text_config``)."""
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def family(config: dict[str, Any]) -> str | None:
    """Which per-layer layout this module reads, or None."""
    text = text_config(config)
    model_type = str(text.get("model_type") or config.get("model_type") or "")
    if model_type.startswith("gemma4") and text.get("layer_types"):
        return "gemma4"
    # DeepSeek V4 compressed + sliding MLA (models/deepseek_v4/attention.py:227-230);
    # V4.1 keeps its compressed caches on source layers only
    # (models/deepseek_v41/attention.py:255-303).
    if model_type == "deepseek_v4" and all(
        text.get(k) for k in ("compress_ratios", "index_head_dim", "sliding_window")
    ):
        return "deepseek_v4"
    if model_type in ("deepseek_v41", "deepseek_v41_text") and all(
        text.get(k)
        for k in (
            "compress_ratios",
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "index_head_dim",
            "sliding_window",
        )
    ):
        return "deepseek_v41"
    # Qwen4Exp: gated delta net + QSA attention with an index side cache + a PLE
    # short conv (models/qwen4_exp/nvidia/model.py:814-835, qsa_cache.py:841-873).
    if model_type.startswith("qwen4_exp") and all(
        text.get(k)
        for k in ("linear_num_value_heads", "layer_types", "indexer_n_heads", "ple_layer_ids")
    ):
        return "qwen4_exp"
    # GLM5Next: KDA + sparse MLA with a kpool indexer (models/glm5next/nvidia/attention.py).
    if (
        model_type.startswith("glm5_next")
        and "deepseek_sparse_attention" in (text.get("layer_types") or ())
        and int(text.get("index_kpool") or 0) > 1
        and text.get("kv_lora_rank")
    ):
        return "glm5_next"
    if model_type.startswith("qwen3_5") and text.get("linear_num_value_heads"):
        return "qwen3_5"
    if model_type == "nemotron_h" and nemotron_pattern(text):
        return "nemotron_h"
    # Sliding + full attention with one head shape (vLLM muse_glimmer.py:1190:
    # "Full attention on NoPE layers, sliding window otherwise").  Other models
    # of this shape (gpt-oss) keep their recorded mechanism until re-recorded.
    if model_type.startswith("muse_glimmer") and set(text.get("layer_types") or ()) == {
        "sliding_attention",
        "full_attention",
    }:
        return "mixed_sliding"
    return None


def kv_layout(config: dict[str, Any]) -> str:
    """How vLLM groups the family's caches into blocks (``bytes_per_sequence``).

    "packed": the block-outermost grouping (kv_cache_utils.py:1978-2112),
    taken when every backend declares a block-outermost layout
    (v1/attention/backends/utils.py:204-236): the DeepSeek V4 indexer
    (v1/attention/backends/mla/indexer.py:245-248) and QSA's state cache
    (models/qwen4_exp/common/qsa_cache.py:753-756).  "glm5_next": GLM-5.3's own
    grouping, tried before it (kv_cache_utils.py:2287, 1195-1275).  "grouped":
    the v0.29 rule of the module docstring.
    """
    name = family(config)
    if name in ("deepseek_v4", "deepseek_v41", "qwen4_exp"):
        return "packed"
    if name == "glm5_next":
        return "glm5_next"
    return "grouped"


# transformers' NemotronHConfig reads ``layers_block_type`` (newer checkpoints,
# e.g. Nemotron-3.5) and hands vLLM the equivalent ``hybrid_override_pattern``
# (configuration_nemotron_h.py ``_list_to_pattern``, legacy names remapped by
# configuration_utils.py ``remap_legacy_layer_types``).
_NEMOTRON_BLOCKS = {
    "mamba": "M",
    "linear_attention": "M",
    "attention": "*",
    "full_attention": "*",
    "mlp": "-",
    "moe": "E",
}


def nemotron_pattern(text: dict[str, Any]) -> str | None:
    """NemotronH's layer pattern, from either config form; None if absent or
    it names a block type transformers does not."""
    if text.get("hybrid_override_pattern"):
        return str(text["hybrid_override_pattern"])
    blocks = text.get("layers_block_type")
    if not isinstance(blocks, list) or not blocks:
        return None
    if any(b not in _NEMOTRON_BLOCKS for b in blocks):
        return None
    return "".join(_NEMOTRON_BLOCKS[b] for b in blocks)


def _heads_per_gpu(heads: int, tp: int) -> int:
    return max(1, -(-int(heads) // tp))


def _ssm_bytes(text: dict[str, Any], key: str) -> int:
    return _DTYPE_BYTES.get(str(text.get(key) or "float32"), 4)


def layer_kinds(
    config: dict[str, Any],
    *,
    tp: int,
    kv_dtype_bytes: int,
    model_dtype_bytes: int,
    sm: int | None = None,
    kv_cache_dtype: str = "auto",
) -> list[LayerKind] | None:
    """Cache-holding layer kinds of a recognised model, or None.

    ``sm`` is the GPU's compute capability as major * 10 + minor (90 for
    H100/H200) and ``kv_cache_dtype`` the engine's ``--kv-cache-dtype``; the
    v0.30 layouts depend on them (the DeepSeek record and block, GLM-5.3's
    backend) and are None where they were not traced.
    """
    text = text_config(config)
    name = family(config)
    if name == "deepseek_v4":
        return _deepseek_v4_kinds(text, sm=sm, kv_cache_dtype=kv_cache_dtype)
    if name == "deepseek_v41":
        return _deepseek_v41_kinds(text, sm=sm, kv_cache_dtype=kv_cache_dtype)
    if name == "qwen4_exp":
        return _qwen4_exp_kinds(
            config, tp=tp, model_dtype_bytes=model_dtype_bytes, kv_cache_dtype=kv_cache_dtype
        )
    if name == "glm5_next":
        return _glm5_next_kinds(
            text, tp=tp, model_dtype_bytes=model_dtype_bytes, sm=sm, kv_cache_dtype=kv_cache_dtype
        )
    if name == "gemma4":
        if int(text.get("num_kv_shared_layers") or 0):
            return None  # shared KV layers: not traced
        types = list(text["layer_types"])
        hd = int(text["head_dim"])
        global_hd = int(text.get("global_head_dim") or hd)
        global_kv = (
            int(text.get("num_global_key_value_heads") or text["num_key_value_heads"])
            if text.get("attention_k_eq_v")
            else int(text["num_key_value_heads"])
        )
        sliding_kv = int(text["num_key_value_heads"])
        full = sum(1 for t in types if t == "full_attention")
        sliding = sum(1 for t in types if t == "sliding_attention")
        return [
            LayerKind(
                "full",
                full,
                bytes_per_token=_heads_per_gpu(global_kv, tp) * 2 * global_hd * kv_dtype_bytes,
            ),
            LayerKind(
                "sliding",
                sliding,
                bytes_per_token=_heads_per_gpu(sliding_kv, tp) * 2 * hd * kv_dtype_bytes,
                window=int(text["sliding_window"]),
            ),
        ]
    if name == "mixed_sliding":
        types = list(text["layer_types"])
        per_token = (
            _heads_per_gpu(int(text["num_key_value_heads"]), tp)
            * 2
            * int(text["head_dim"])
            * kv_dtype_bytes
        )
        return [
            LayerKind("full", types.count("full_attention"), bytes_per_token=per_token),
            LayerKind(
                "sliding",
                types.count("sliding_attention"),
                bytes_per_token=per_token,
                window=int(text["sliding_window"]),
            ),
        ]
    if name == "qwen3_5":
        n = int(text["num_hidden_layers"])
        types = text.get("layer_types") or [
            "full_attention"
            if (i + 1) % int(text.get("full_attention_interval") or 4) == 0
            else "linear_attention"
            for i in range(n)
        ]
        full = sum(1 for t in types if t == "full_attention")
        linear = sum(1 for t in types if t == "linear_attention")
        k_heads, v_heads = int(text["linear_num_key_heads"]), int(text["linear_num_value_heads"])
        k_dim, v_dim = int(text["linear_key_head_dim"]), int(text["linear_value_head_dim"])
        conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
        conv = (int(text["linear_conv_kernel_dim"]) - 1) * (conv_dim // tp) * model_dtype_bytes
        temporal = (v_heads // tp) * v_dim * k_dim * _ssm_bytes(text, "mamba_ssm_dtype")
        return [
            LayerKind(
                "full",
                full,
                bytes_per_token=_heads_per_gpu(int(text["num_key_value_heads"]), tp)
                * 2
                * int(text["head_dim"])
                * kv_dtype_bytes,
            ),
            LayerKind("state", linear, state_bytes=conv + temporal),
        ]
    if name == "nemotron_h":
        pattern = nemotron_pattern(text) or ""
        heads, hd = int(text["mamba_num_heads"]), int(text["mamba_head_dim"])
        state, groups = int(text["ssm_state_size"]), int(text["n_groups"])
        ng = groups if groups % tp == 0 else groups + (tp - groups)
        conv = (
            (int(text["conv_kernel"]) - 1)
            * ((heads * hd + 2 * ng * state) // tp)
            * model_dtype_bytes
        )
        temporal = (heads // tp) * hd * state * _ssm_bytes(text, "mamba_ssm_cache_dtype")
        head_dim = int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"])
        return [
            LayerKind(
                "full",
                pattern.count("*"),
                bytes_per_token=_heads_per_gpu(int(text["num_key_value_heads"]), tp)
                * 2
                * head_dim
                * kv_dtype_bytes,
            ),
            LayerKind("state", pattern.count("M"), state_bytes=conv + temporal),
        ]
    return None


def _aligned(raw: int, alignment: int) -> int:
    """A page rounded up to its alignment (kv_cache_interface.py:636-642)."""
    return -(-raw // alignment) * alignment


# DeepSeek's fp8_ds_mla record: 448 fp8 NoPE + 64 bf16 RoPE + 7 UE8M0 scales +
# 1 pad = 584 bytes per state, pages rounded to FlashMLA's 576-byte stride
# (models/deepseek_v4/attention.py:857-862; v1/attention/backends/mla/
# sparse_swa.py:82-83; models/deepseek_v41/attention.py:115-117,459-461).
_DS_MLA_RECORD = 584
_DS_MLA_ALIGN = 576


def _deepseek_fp8_kv(kv_cache_dtype: str) -> bool:
    """FlashMLA's fp8_ds_mla layout takes auto or any fp8 KV dtype and refuses
    the rest (models/deepseek_v4/attention.py:94-123)."""
    return kv_cache_dtype == "auto" or kv_cache_dtype.startswith("fp8")


def _deepseek_indexer_row(text: dict[str, Any]) -> int:
    """fp8 indexer key: head_dim fp8 bytes + one fp32 scale per 128
    (models/deepseek_v4/attention.py:990-997; the MXFP4 indexer is opt-in,
    v1/attention/backends/mla/indexer.py:58-73)."""
    ihd = int(text["index_head_dim"])
    return ihd + ihd // 128 * 4


def _deepseek_v4_kinds(
    text: dict[str, Any], *, sm: int | None, kv_cache_dtype: str
) -> list[LayerKind] | None:
    """DeepSeek V4 on FlashMLA: every GPU keeps the whole cache (one KV head).

    SM 9.x/10.x select FlashMLA; SM 12.x FlashInfer with other pages
    (models/deepseek_v4/nvidia/model.py:1085-1117), not traced.  Blocks: 256
    tokens for the compressed caches (sparse_mla.py:59-61, sparse_swa.py:148-149,
    platforms/interface.py:629-640), 64 for the sliding-window cache
    (sparse_swa.py:81), 4 / 8 for the ratio-4 / ratio-128 compressor states
    (compressor.py:158-171).
    """
    if sm is None or sm // 10 not in (9, 10) or not _deepseek_fp8_kv(kv_cache_dtype):
        return None
    n = int(text["num_hidden_layers"])
    # MTP layers are past the list's first n entries (attention.py:224-230).
    ratios = [max(1, int(r)) for r in list(text["compress_ratios"])[:n]]
    if len(ratios) != n or set(ratios) - {1, 4, 128}:
        return None
    c4, c128 = ratios.count(4), ratios.count(128)
    hd, ihd = int(text["head_dim"]), int(text["index_head_dim"])
    block = 256

    def mla(states: int, content: int) -> int:
        return _aligned(states * content, _DS_MLA_ALIGN)

    return [
        # Sliding-window MLA cache on every layer (sparse_swa.py:111-133).
        LayerKind(
            "sliding",
            n,
            window=int(text["sliding_window"]),
            block_size=64,
            page_bytes=mla(64, _DS_MLA_RECORD),
        ),
        # Compressed MLA, one state per ratio tokens (attention.py:840-862).
        LayerKind("full", c4, block_size=block, page_bytes=mla(block // 4, _DS_MLA_RECORD)),
        LayerKind("full", c128, block_size=block, page_bytes=mla(block // 128, _DS_MLA_RECORD)),
        # Indexer keys on ratio-4 layers (attention.py:292-311, 891-902).
        LayerKind(
            "compressed",
            c4,
            block_size=block,
            page_bytes=mla(block // 4, _deepseek_indexer_row(text)),
        ),
        # Compressor states: fp32 [kv, score] x (1 + overlap) per state, a
        # sliding window of (1 + overlap) * ratio (compressor.py:156-195, 270-284).
        LayerKind("sliding", c4, window=8, block_size=4, page_bytes=mla(4, 2 * 2 * hd * 4)),
        LayerKind("sliding", c128, window=128, block_size=8, page_bytes=mla(8, 2 * hd * 4)),
        LayerKind("sliding", c4, window=8, block_size=4, page_bytes=mla(4, 2 * 2 * ihd * 4)),
    ]


def _deepseek_v41_kinds(
    text: dict[str, Any], *, sm: int | None, kv_cache_dtype: str
) -> list[LayerKind] | None:
    """DeepSeek V4.1 on FlashMLA, SM 9.x: compressed caches only on
    ``kv_source_layer_ids``, indexer keys on those that are also index sources
    (models/deepseek_v41/attention.py:255-303, 380-400).

    SM 10.x writes a 528-byte record in 512-byte pages on 128-token blocks
    (attention.py:115-117, sparse_mla.py:88-90): not traced.  Blocks: 64 for
    the compressed caches (sparse_mla.py:88-90, indexer.py:261-263), 32 for
    the sliding-window cache (attention.py:463-473), 8 for the compressor's ring
    (compressor.py:132-163).
    """
    if sm is None or sm // 10 != 9 or not _deepseek_fp8_kv(kv_cache_dtype):
        return None
    n = int(text["num_hidden_layers"])
    ratios = list(text["compress_ratios"])
    index_sources = {int(i) for i in text["index_source_layer_ids"]}
    hd = int(text["head_dim"])
    block = 64
    pages: Counter[tuple[str, int, int]] = Counter()
    for i in (int(i) for i in text["kv_source_layer_ids"]):
        if i >= n:
            continue  # MTP layers are not backbone layers (attention.py:284-286)
        if i >= len(ratios) or int(ratios[i]) not in (1, 2):
            return None
        r = int(ratios[i])
        # Compressed MLA (attention.py:947-972) and indexer keys (1009-1026).
        pages["full", block, _aligned(block // r * _DS_MLA_RECORD, _DS_MLA_ALIGN)] += 1
        if i in index_sources:
            row = _deepseek_indexer_row(text)
            pages["compressed", block, _aligned(block // r * row, _DS_MLA_ALIGN)] += 1
        if r > 1:
            # One ring of fp32 [kv, score] rows per request, no alignment
            # (compressor.py:132-163, 230-238).
            pages["ring", 8, 8 * 2 * hd * 4] += 1
    return [
        LayerKind(
            "sliding",
            n,
            window=int(text["sliding_window"]),
            block_size=32,
            page_bytes=_aligned(32 * _DS_MLA_RECORD, _DS_MLA_ALIGN),
        ),
        *(
            LayerKind(kind, count, block_size=size, page_bytes=page)
            for (kind, size, page), count in pages.items()
        ),
    ]


def _qwen4_exp_kinds(
    config: dict[str, Any], *, tp: int, model_dtype_bytes: int, kv_cache_dtype: str
) -> list[LayerKind] | None:
    """Qwen4Exp: full attention (QSA), gated delta net, one PLE short conv.

    QSA refuses a KV cache other than bf16 (models/qwen4_exp/nvidia/qsa.py:
    186-189).  The block is the smallest 16-token multiple whose attention page
    holds the largest state (platforms/interface.py:866-939, alignment 16 from
    qsa.py:74-76); both state kinds are padded to that page.
    """
    if kv_cache_dtype not in ("auto", "bfloat16"):
        return None
    text = text_config(config)
    types = list(text["layer_types"])
    full, linear = types.count("full_attention"), types.count("linear_attention")
    k_heads, v_heads = int(text["linear_num_key_heads"]), int(text["linear_num_value_heads"])
    k_dim, v_dim = int(text["linear_key_head_dim"]), int(text["linear_value_head_dim"])
    conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
    if conv_dim % tp or v_heads % tp:
        return None
    # mamba_utils.py:274-295; fp32 temporal state from mamba_ssm_dtype
    # (model_executor/models/config.py:801-815).
    gdn = (int(text["linear_conv_kernel_dim"]) - 1) * (conv_dim // tp) * model_dtype_bytes + (
        v_heads // tp
    ) * v_dim * k_dim * _ssm_bytes(text, "mamba_ssm_dtype")
    # PLE short conv over the hyper-connection width, dilated by ngram_size,
    # every rank the whole state (nvidia/model.py:722-743, 828-833).
    ple = (
        (int(text["ple_conv_kernel_size"]) - 1)
        * int(text["ngram_size"])
        * int(text["hidden_size"])
        * int(text["hc_count"])
        * model_dtype_bytes
    )
    per_token = (
        _heads_per_gpu(int(text["num_key_value_heads"]), tp) * 2 * int(text["head_dim"]) * 2
    )
    block = BASE_BLOCK * ceil(max(gdn, ple) / (BASE_BLOCK * per_token))
    page = block * per_token
    ratio, index_dim = int(text["indexer_compress_ratio"]), int(text["indexer_head_dim"])
    if block % ratio:
        return None  # vLLM refuses it (common/qsa_cache.py:777-780)
    # The raw-key ring carries three int64 M-RoPE positions per row unless
    # vLLM strips the M-RoPE fields: always for the text-only architecture,
    # otherwise only with language_model_only (models/config.py:842-887, 905).
    architecture = (config.get("architectures") or [""])[0]
    mrope = bool((text.get("rope_parameters") or {}).get("mrope_section")) and (
        architecture != "Qwen4ExpForCausalLM"
    )
    ring_row = (-(-index_dim // 4) * 4 + 3 * 4) if mrope else index_dim
    return [
        LayerKind("full", full, bytes_per_token=per_token, block_size=block, page_bytes=page),
        LayerKind(
            "state", linear, state_bytes=gdn, block_size=block, page_bytes=page, bucket="gdn"
        ),
        LayerKind(
            "state",
            len(text["ple_layer_ids"]),
            state_bytes=ple,
            block_size=block,
            page_bytes=page,
            bucket="ple",
        ),
        # Compressed index key, one bf16 row per ratio tokens on the attention
        # block (common/qsa_cache.py:862-873; bf16 by default, indexer_qsa.py:152-164).
        LayerKind("compressed", full, block_size=block, page_bytes=block // ratio * index_dim * 2),
        # Raw-key ring of ``ratio`` bf16 rows (common/qsa_cache.py:806-860).
        LayerKind("ring", full, block_size=ratio, page_bytes=ratio * ring_row * 2),
    ]


def _glm5_next_kinds(
    text: dict[str, Any],
    *,
    tp: int,
    model_dtype_bytes: int,
    sm: int | None,
    kv_cache_dtype: str,
) -> list[LayerKind] | None:
    """GLM5Next: KDA state + sparse MLA latent + kpool indexer keys + a tail.

    With a bf16 or ``auto`` KV dtype the MLA latent stays bf16 except on SM 12.x,
    where FlashInfer's SM120 backend switches to fp8_ds_mla
    (model_executor/layers/attention/mla_attention.py:359-376), and the kpool
    pool page is 64 there (platforms/cuda.py:429-447): not traced.  A fp8 KV
    cache is fp8_ds_mla on FlashMLA and plain fp8 on FlashInfer (365-369): not
    traced either.
    """
    if kv_cache_dtype not in ("auto", "bfloat16") or sm is None or sm // 10 == 12:
        return None
    types = list(text["layer_types"])
    dsa, kda = types.count("deepseek_sparse_attention"), types.count("linear_attention")
    # vLLM's config reads the KDA shape from linear_attn_config first
    # (transformers_utils/configs/glm5_next.py:99-110).
    linear = text.get("linear_attn_config") or {}
    heads = int(linear.get("num_heads") or text.get("linear_num_heads") or 64)
    head_dim = int(linear.get("head_dim") or text.get("linear_head_dim") or 128)
    kernel = int(linear.get("short_conv_kernel_size") or text.get("linear_conv_kernel_dim") or 4)
    if heads % tp:
        return None  # kda.py:199
    # Conv (K - 1, 3 * H * D / tp) in the model dtype, recurrent (H / tp, D, D)
    # fp32 whatever --mamba-ssm-cache-dtype says (glm5next/nvidia/kda.py:158-180,
    # mamba_utils.py:133-149, 297-321).
    state = (kernel - 1) * (3 * heads * head_dim // tp) * model_dtype_bytes + (
        heads // tp
    ) * head_dim * head_dim * 4
    # MLA latent: one bf16 head of kv_lora_rank + rope on every rank
    # (mla_attention.py:462, 1347-1374).
    per_token = (int(text["kv_lora_rank"]) + int(text.get("qk_rope_head_dim") or 0)) * 2
    # Block: MLA hybrids align to 128 tokens (platforms/interface.py:894-907,
    # 923-926), then to index_kpool * 32 (interface.py:927-930, cuda.py:429-447,
    # utils/deep_gemm.py:35).
    kpool = int(text["index_kpool"])
    block = 128 * ceil(state / (128 * per_token))
    block = kpool * 32 * ceil(block / (kpool * 32))
    page = block * per_token
    index_dim = int(text["index_head_dim"])
    return [
        LayerKind("full", dsa, bytes_per_token=per_token, block_size=block, page_bytes=page),
        # fp8 keys + one fp32 scale per 128, one row per kpool tokens
        # (glm5next/nvidia/attention.py:77-157, 279-285).
        LayerKind(
            "compressed",
            dsa,
            block_size=block,
            page_bytes=block // kpool * (index_dim + index_dim // 128 * 4),
        ),
        # One tail block per request (attention.py:159-208).
        LayerKind("tail", dsa, block_size=kpool, page_bytes=kpool * 2 * index_dim * 2),
        LayerKind("state", kda, state_bytes=state, block_size=block, page_bytes=page),
    ]


def _approximate_gcd(values: list[int], lower_bound: int) -> int:
    """The repeat count with the least padding, ties to the larger
    (kv_cache_utils.py:1943-1975)."""
    low = max(1, lower_bound)
    if low > max(values):
        return low
    best, best_pad = low, None
    for d in range(low, max(values) + 1):
        pad = sum((d - x % d) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best):
            best, best_pad = d, pad
    return best


def _blocks_per_request(
    kind: LayerKind, max_model_len: int, in_flight_tokens: int, prefix_caching: bool
) -> int:
    """Pages one request holds in one group of this kind."""
    if kind.kind in ("full", "compressed"):
        return ceil(max_model_len / kind.block_size)  # kv_cache_interface.py:564-569
    if kind.kind == "sliding":
        # kv_cache_interface.py:816-842
        span = min(kind.window - 1 + in_flight_tokens, max_model_len)
        return ceil(span / kind.block_size) + 1
    if kind.kind == "state":
        return 2 if prefix_caching else 1  # kv_cache_interface.py:1034-1045
    return 1  # ring, tail: one block for the request's lifetime (879-882, 971-990)


class Blocks(NamedTuple):
    """vLLM v0.30's pool accounting for one request: every group takes whole
    blocks of ``bytes_per_block`` from the shared pool."""

    bytes_per_block: int  # the widest group's pages (kv_cache_utils.py:1588-1611)
    blocks: int  # one request's blocks, all groups together
    groups: int  # KV cache groups

    @property
    def bytes_per_sequence(self) -> int:
        return self.bytes_per_block * self.blocks


def _packed_blocks(
    kinds: list[LayerKind], max_model_len: int, in_flight_tokens: int, prefix_caching: bool
) -> Blocks | None:
    """Block-outermost grouping (kv_cache_utils.py:1978-2112) and one
    request's reservation (kv_cache_utils.py:1588-1611, 2448-2461).

    Layers are bucketed by vLLM's uniform-type rule (same block and spec
    family; sliding windows and Mamba pages must match, kv_cache_interface.py:
    1233-1246); a bucket with one layer per page size per repeat is split
    into groups of a common repeat count, a state bucket down to what a block
    already holds.  A block is the widest group's pages; a request takes the
    sum of its groups' pages, block for block.
    """
    buckets: dict[tuple[str, int, int, str], list[LayerKind]] = {}
    for k in kinds:
        family_ = "full" if k.kind in ("full", "compressed") else k.kind
        if family_ not in ("full", "sliding", "ring", "state") or k.block_size <= 0:
            return None
        key = (
            family_,
            k.block_size,
            k.window if family_ == "sliding" else (k.page_bytes if family_ == "state" else 0),
            k.bucket,
        )
        buckets.setdefault(key, []).append(k)
    if len({k.page_bytes for k in kinds}) <= 1:
        return None  # one page size: vLLM does not pack (kv_cache_utils.py:1995-1996)
    shaped = []
    for members in buckets.values():
        pages: Counter[int] = Counter()
        for k in members:
            pages[k.page_bytes] += k.count
        blocks = {
            _blocks_per_request(k, max_model_len, in_flight_tokens, prefix_caching)
            for k in members
        }
        balanced = len(set(pages.values())) == 1
        shaped.append((members[0].kind, pages, balanced, max(blocks)))
    floor = max((max(p.values()) for _, p, bal, _ in shaped if bal and len(p) > 1), default=0)
    repeats = (
        _approximate_gcd([max(p.values()) for _, p, bal, _ in shaped if bal], floor)
        if floor
        else None
    )

    def is_state(kind: str) -> bool:
        return kind in ("state", "ring") or (repeats is None and kind == "sliding")

    def groups_for(pages: Counter[int], balanced: bool) -> int:
        return ceil(max(pages.values()) / repeats) if balanced and repeats else 1

    def widest(pages: Counter[int], n: int) -> int:
        return sum(ceil(count / n) * page for page, count in pages.items())

    anchor = max(
        widest(p, sum(p.values()) if is_state(kind) else groups_for(p, bal))
        for kind, p, bal, _ in shaped
    )
    bytes_per_block = blocks = groups = 0
    for kind, pages, balanced, per_group in shaped:
        n = groups_for(pages, balanced)
        if is_state(kind):
            if len(pages) != 1:
                return None  # a state bucket of mixed pages: not traced
            n = max(n, ceil(sum(pages.values()) / max(anchor // next(iter(pages)), 1)))
        if n > 1 and not balanced:
            return None  # an unbalanced bucket is emitted whole; not split
        bytes_per_block = max(bytes_per_block, widest(pages, n))
        blocks += n * per_group
        groups += n
    return Blocks(bytes_per_block, blocks, groups)


def _glm5_next_blocks(
    kinds: list[LayerKind], max_model_len: int, prefix_caching: bool
) -> Blocks | None:
    """GLM5Next's grouping (kv_cache_utils.py:1195-1275) and reservation
    (2425-2446): one group of every MLA and indexer page, the KDA layers in
    ``ceil(kda / mla)`` groups padded to the MLA page, the tails in one group
    that costs one block.  A block holds every MLA and indexer page
    (1592-1594)."""
    by_kind: dict[str, list[LayerKind]] = {}
    for k in kinds:
        by_kind.setdefault(k.kind, []).append(k)
    if sorted(by_kind) != ["compressed", "full", "state", "tail"] or any(
        len(v) != 1 for v in by_kind.values()
    ):
        return None
    (full,), (index,), (state,) = by_kind["full"], by_kind["compressed"], by_kind["state"]
    if state.state_bytes > full.page_bytes:
        return None  # vLLM refuses it (kv_cache_utils.py:1249-1254)
    per_block = full.count * full.page_bytes + index.count * index.page_bytes
    state_groups = ceil(state.count / full.count)  # pipeline parallel 1 (1172-1175)
    blocks = ceil(max_model_len / full.block_size) + state_groups * (2 if prefix_caching else 1)
    return Blocks(per_block, blocks + 1, 2 + state_groups)


def _grouped_blocks(
    kinds: list[LayerKind], max_model_len: int, in_flight_tokens: int, prefix_caching: bool
) -> Blocks | None:
    """vLLM v0.29's grouping (the module docstring): a block is ``group_size``
    pages of one size (kv_cache_utils.py:1358-1370); one request takes, per
    group, its kind's pages."""
    attention = [k for k in kinds if k.kind != "state"]
    states = [k for k in kinds if k.kind == "state"]
    blocks: dict[int, int] = {}  # id(kind) -> block size in tokens
    if states:
        if len(states) != 1 or len({k.bytes_per_token for k in attention}) != 1:
            return None  # one state kind beside one attention size: what was traced
        per_token = attention[0].bytes_per_token
        block = BASE_BLOCK * ceil(states[0].state_bytes / (BASE_BLOCK * per_token))
        page = block * per_token
        for k in attention:
            blocks[id(k)] = block
    else:
        page = max(BASE_BLOCK * k.bytes_per_token for k in attention)
        for k in attention:
            if page % k.bytes_per_token:
                return None  # pages that do not divide: not traced
            blocks[id(k)] = page // k.bytes_per_token
    counts = [k.count for k in kinds]
    smallest, largest = min(counts), max(counts)
    group_size = largest if largest < 1.5 * smallest else smallest
    per_request = groups_total = 0
    for k in kinds:
        groups = ceil(k.count / group_size)
        if k.kind == "state":
            per_group = 2 if prefix_caching else 1
        elif k.kind == "sliding":
            span = min(k.window - 1 + in_flight_tokens, max_model_len)
            per_group = ceil(span / blocks[id(k)]) + 1
        else:
            per_group = ceil(max_model_len / blocks[id(k)])
        per_request += groups * per_group
        groups_total += groups
    return Blocks(group_size * page, per_request, groups_total)


def block_accounting(
    kinds: list[LayerKind],
    max_model_len: int,
    *,
    in_flight_tokens: int,
    prefix_caching: bool = True,
    layout: str,
) -> Blocks | None:
    """Bytes per block, blocks and groups of one request under ``layout``
    (``kv_layout``: "grouped", "packed" or "glm5_next"); None when the layout
    or the kinds were not traced.  vLLM sizes the pool as
    ``available // bytes_per_block`` blocks (v0.29 kv_cache_utils.py:1440,
    v0.30 kv_cache_utils.py:1695, 1750)."""
    kinds = [k for k in kinds if k.count > 0]
    if not kinds:
        return None
    if layout == "grouped":
        return _grouped_blocks(kinds, max_model_len, in_flight_tokens, prefix_caching)
    if layout == "packed":
        return _packed_blocks(kinds, max_model_len, in_flight_tokens, prefix_caching)
    if layout == "glm5_next":
        return _glm5_next_blocks(kinds, max_model_len, prefix_caching)
    return None


def bytes_per_sequence(
    kinds: list[LayerKind],
    max_model_len: int,
    *,
    in_flight_tokens: int,
    prefix_caching: bool = True,
    layout: str = "grouped",
) -> int | None:
    """What one request of ``max_model_len`` tokens reserves, as vLLM counts it.

    ``layout`` is ``kv_layout(config)``: "packed" and "glm5_next" are vLLM
    v0.30's groupings for the families whose kinds carry their pages.
    """
    accounted = block_accounting(
        kinds,
        max_model_len,
        in_flight_tokens=in_flight_tokens,
        prefix_caching=prefix_caching,
        layout=layout,
    )
    return None if accounted is None else accounted.bytes_per_sequence


# Decode sequences a state (Mamba) cache can serve.  vLLM needs one state block
# per decode sequence: with full CUDA graphs it refuses ``max_num_seqs`` above
# the pool's blocks (v0.29 config/compilation.py:1505-1520, v0.30 1527-1542),
# and its CUDA-graph profiling allocates ``min(max_num_seqs,
# max_cudagraph_capture_size)`` blocks before the real cache (v0.29
# v1/worker/gpu_model_runner.py:6598-6602, v0.30 v1/worker/gpu/
# cudagraph_utils.py:972-976).  The pool is ``available // bytes_per_block``.
#
# The safety buffer: available KV memory is predicted, not measured.  Over
# every healthy cohort memory record (28 boots, vLLM v0.29.0 and v0.30.0, RTX
# 4090 to H100; ``_dev_notes/cohort-run/kv-budget-residuals.md``,
# ``scripts/kv_budget_residuals.py``) the prediction's largest excess over
# vLLM's figure, once a held compile segment is set apart, is
# KV_BUDGET_SAFETY_BUFFER_BYTES: GLM-4.7-Flash on an H100, whose MLA CUDA-graph
# estimate (3.25 GiB) the per-layer rule does not model.  The buffer is that
# measured error, not a fitted margin; ``test_calculator_vs_cohort`` fails when
# a record exceeds it or when it exceeds the largest error.  A compile segment
# vLLM may hold (``calculator.compile_segment_bytes``, up to V x H x b on a
# text-only TP 1 plan) is subtracted on top, per plan.
# 2,340,479,996 B measured (2.18 GiB), plus the startup log's 0.005 GiB
# rounding of the measured figure, rounded up to 0.01 GiB.
KV_BUDGET_SAFETY_BUFFER_BYTES = 2_351_494_595  # 2.19 GiB


def state_block_capacity(
    available_kv_bytes: int,
    bytes_per_block: int,
    *,
    margin_bytes: int = KV_BUDGET_SAFETY_BUFFER_BYTES,
) -> int:
    """KV blocks vLLM is predicted to allocate at the least: the predicted
    available memory less *margin_bytes* (the safety buffer, plus a compile
    segment where one may be held), in whole blocks; at least 0."""
    if bytes_per_block <= 0:
        return 0
    return max(0, (int(available_kv_bytes) - margin_bytes) // int(bytes_per_block))


def duplicated_weight_bytes(config: dict[str, Any], *, tp: int, dtype_bytes: int) -> int:
    """Weights vLLM creates that the checkpoint does not store.

    Gemma 4 with ``attention_k_eq_v``: full-attention layers have no v_proj;
    vLLM copies k_proj into a v_proj slot (``models/gemma4.py:1711-1716``).
    """
    text = text_config(config)
    if family(config) != "gemma4" or not text.get("attention_k_eq_v"):
        return 0
    full = sum(1 for t in text["layer_types"] if t == "full_attention")
    kv = _heads_per_gpu(int(text.get("num_global_key_value_heads") or 1), tp)
    hd = int(text.get("global_head_dim") or text["head_dim"])
    return full * kv * hd * int(text["hidden_size"]) * dtype_bytes


def padded_weight_bytes(config: dict[str, Any], *, tp: int) -> int:
    """Per-GPU bytes vLLM allocates beyond the checkpoint's shard: DeepSeek V4 /
    V4.1's MXFP4 experts, whose per-rank intermediate size every CUDA backend
    rounds up to a multiple of 128.

    fp4 experts take ``Mxfp4MoEMethod`` (models/deepseek_v4/quant_config.py:
    173-191), whose backend is one of TRTLLM MXFP8, DeepGEMM or Marlin
    (layers/fused_moe/oracle/mxfp4.py:357-381, 648-716); each rounds the
    intermediate size to 128 (oracle/mxfp4.py:740-752), applied before the
    weights are created (layers/fused_moe/routed_experts.py:128-141).  Per
    expert and layer, ``w13`` holds 2 x I x H / 2 bytes and H / 32 scales per
    row, ``w2`` H x I / 2 bytes and I / 32 scales per row
    (layers/quantization/mxfp4.py:566-634).  Hidden sizes 4096 and 5120 need
    no rounding.  V4.1 at TP 4: 2304 / 4 = 576 -> 640 per rank.
    """
    text = text_config(config)
    if family(config) not in ("deepseek_v4", "deepseek_v41"):
        return 0
    quant = config.get("quantization_config") or text.get("quantization_config") or {}
    expert_dtype = text.get("expert_dtype") or quant.get("expert_dtype") or "fp4"
    if expert_dtype != "fp4" or quant.get("moe_quant_algo"):
        return 0
    per_rank = int(text["moe_intermediate_size"]) // tp
    extra = -(-per_rank // 128) * 128 - per_rank
    hidden = int(text["hidden_size"])
    layers = int(text["num_hidden_layers"]) - int(text.get("first_k_dense_replace") or 0)
    experts = int(text["n_routed_experts"])
    return (
        layers
        * experts
        * extra
        * (2 * hidden // 2 + 2 * hidden // 32 + hidden // 2 + hidden // 32)
    )
