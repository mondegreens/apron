"""Dense GQA KV cache calculator for autoregressive_decode mechanism.

Mechanism-aware memory prediction for transformer models with grouped-query
attention. Produces a PlanningClaim with 7 breakdown fields and
EpistemicStatus: predicted with wide uncertainty bounds (uncalibrated).
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
    """Extract GQA parameters from config.json-style metadata.

    Returns None if required fields are missing.
    """
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


@register_calculator("autoregressive_decode")
def calculate_autoregressive_decode(inputs: CalculatorInput) -> dict[str, Any] | None:
    """Dense GQA memory prediction for autoregressive decode models."""
    metadata = inputs.artifact_metadata
    gqa = _extract_gqa_params(metadata)
    if gqa is None:
        return None

    torch_dtype = metadata.get("torch_dtype", "bfloat16")
    dtype_bytes = DTYPE_BYTES.get(torch_dtype, 2)

    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None

    isl = 512
    osl = 128
    max_batch_size = 4
    workload = inputs.workload
    if workload.kind == "text":
        isl = workload.input_length  # type: ignore[union-attr]
        osl = workload.output_length  # type: ignore[union-attr]
    max_batch_size = inputs.execution_spec_data.get("max_batch_size", max_batch_size)

    kv_per_token = 2 * gqa["num_layers"] * gqa["num_kv_heads"] * gqa["head_dim"] * dtype_bytes
    kv_cache = kv_per_token * (isl + osl) * max_batch_size

    activation_estimate = int(weight_bytes * 0.10)
    non_pytorch_overhead = 500_000_000
    enforce_eager = inputs.execution_spec_data.get("enforce_eager", False)
    cuda_graph_estimate = 0 if enforce_eager else 800_000_000

    total_required = (
        weight_bytes + kv_cache + activation_estimate + non_pytorch_overhead + cuda_graph_estimate
    )

    gpu_memory_utilization = inputs.execution_spec_data.get("gpu_memory_utilization", 0.90)
    gpu_available = int(inputs.hardware.total_memory_bytes * gpu_memory_utilization)
    available_kv_cache = (
        gpu_available
        - weight_bytes
        - activation_estimate
        - non_pytorch_overhead
        - cuda_graph_estimate
    )

    return {
        "weight_memory_bytes": weight_bytes,
        "kv_cache_bytes": kv_cache,
        "kv_per_token_bytes": kv_per_token,
        "activation_estimate_bytes": activation_estimate,
        "non_pytorch_overhead_bytes": non_pytorch_overhead,
        "cuda_graph_estimate_bytes": cuda_graph_estimate,
        "total_required_bytes": total_required,
        "available_kv_cache_bytes": available_kv_cache,
        "gpu_available_bytes": gpu_available,
        "num_layers": gqa["num_layers"],
        "num_attention_heads": gqa["num_attention_heads"],
        "num_kv_heads": gqa["num_kv_heads"],
        "head_dim": gqa["head_dim"],
        "dtype_bytes": dtype_bytes,
        "isl": isl,
        "osl": osl,
        "max_batch_size": max_batch_size,
    }
