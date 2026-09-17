"""Mechanism-aware memory calculators.

Each registered branch handles a specific attention/state mechanism:
- autoregressive_decode: dense GQA/MHA/MQA (Llama, Qwen, Gemma, etc.)
- mla_decode: Multi-Latent Attention (DeepSeek-V2/V3/R1)

MoE activation adjustment and sliding-window KV cap are applied
within each branch when the config signals them.
"""

from __future__ import annotations

from typing import Any

from apron.domain.mechanisms import CalculatorInput, register_calculator

DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 1,
}


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


def _adjust_activation_for_moe(activation_estimate: int, metadata: dict[str, Any]) -> int:
    """Scale activation estimate by the MoE expert routing ratio."""
    n_routed = metadata.get("n_routed_experts", 0)
    n_active = metadata.get("num_experts_per_tok", 0)
    if n_routed > 0 and n_active > 0 and n_active < n_routed:
        return int(activation_estimate * n_active / n_routed)
    return activation_estimate


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


def _common_overhead(
    weight_bytes: int,
    kv_cache: int,
    activation_estimate: int,
    inputs: CalculatorInput,
) -> dict[str, int]:
    """Compute overhead fields shared across mechanism branches."""
    non_pytorch_overhead = 500_000_000
    enforce_eager = inputs.execution_spec_data.get("enforce_eager", False)
    cuda_graph_estimate = 0 if enforce_eager else 800_000_000

    total_required = (
        weight_bytes + kv_cache + activation_estimate + non_pytorch_overhead + cuda_graph_estimate
    )

    gpu_utilization = inputs.execution_spec_data.get("gpu_memory_utilization", 0.90)
    gpu_available = int(inputs.hardware.total_memory_bytes * gpu_utilization)
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

    torch_dtype = metadata.get("torch_dtype", "bfloat16")
    dtype_bytes = DTYPE_BYTES.get(torch_dtype, 2)

    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None

    isl, osl, max_batch_size = _workload_params(inputs)
    seq_len = isl + osl

    kv_per_token = 2 * gqa["num_layers"] * gqa["num_kv_heads"] * gqa["head_dim"] * dtype_bytes
    kv_cache, effective_seq = _apply_sliding_window(
        kv_per_token, seq_len, max_batch_size, metadata
    )

    activation_estimate = _adjust_activation_for_moe(int(weight_bytes * 0.10), metadata)

    overhead = _common_overhead(weight_bytes, kv_cache, activation_estimate, inputs)

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

    torch_dtype = metadata.get("torch_dtype", "bfloat16")
    dtype_bytes = DTYPE_BYTES.get(torch_dtype, 2)

    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None

    isl, osl, max_batch_size = _workload_params(inputs)
    seq_len = isl + osl

    cache_per_layer = (mla["kv_lora_rank"] + mla["qk_rope_head_dim"]) * dtype_bytes
    kv_per_token = cache_per_layer * mla["num_layers"]
    kv_cache, effective_seq = _apply_sliding_window(
        kv_per_token, seq_len, max_batch_size, metadata
    )

    activation_estimate = _adjust_activation_for_moe(int(weight_bytes * 0.10), metadata)

    overhead = _common_overhead(weight_bytes, kv_cache, activation_estimate, inputs)

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
        "mechanism_detail": "mla",
    }
