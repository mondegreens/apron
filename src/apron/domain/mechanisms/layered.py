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

Pure arithmetic on config values; no I/O.  Predictions until a boot checks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

BASE_BLOCK = 16  # config/cache.py:79 and the 16-token kernel alignment
_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "fp8": 1, "float8": 1}


@dataclass(frozen=True)
class LayerKind:
    """One kind of cache-holding layer and how many there are."""

    kind: str  # "full", "sliding" or "state"
    count: int
    bytes_per_token: int = 0  # attention: K and V per context token, per GPU
    window: int = 0  # sliding attention
    state_bytes: int = 0  # recurrent state per sequence, per GPU


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
    if model_type.startswith("qwen3_5") and text.get("linear_num_value_heads"):
        return "qwen3_5"
    if model_type == "nemotron_h" and text.get("hybrid_override_pattern"):
        return "nemotron_h"
    return None


def _heads_per_gpu(heads: int, tp: int) -> int:
    return max(1, -(-int(heads) // tp))


def _ssm_bytes(text: dict[str, Any], key: str) -> int:
    return _DTYPE_BYTES.get(str(text.get(key) or "float32"), 4)


def layer_kinds(
    config: dict[str, Any], *, tp: int, kv_dtype_bytes: int, model_dtype_bytes: int
) -> list[LayerKind] | None:
    """Cache-holding layer kinds of a recognised model, or None."""
    text = text_config(config)
    name = family(config)
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
        pattern = str(text["hybrid_override_pattern"])
        heads, hd = int(text["mamba_num_heads"]), int(text["mamba_head_dim"])
        state, groups = int(text["ssm_state_size"]), int(text["n_groups"])
        ng = groups if groups % tp == 0 else groups + (tp - groups)
        conv = (
            (int(text["conv_kernel"]) - 1) * ((heads * hd + 2 * ng * state) // tp) * model_dtype_bytes
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


def bytes_per_sequence(
    kinds: list[LayerKind],
    max_model_len: int,
    *,
    in_flight_tokens: int,
    prefix_caching: bool = True,
) -> int | None:
    """What one request of ``max_model_len`` tokens reserves, as vLLM counts it."""
    kinds = [k for k in kinds if k.count > 0]
    if not kinds:
        return None
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
    per_request = 0
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
    return per_request * group_size * page


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
