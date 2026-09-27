"""Mechanism-aware memory calculators.

Each registered branch handles a specific attention/state mechanism:
- autoregressive_decode: dense GQA/MHA/MQA (Llama, Qwen, Gemma, etc.)
- mla_decode: Multi-Latent Attention (DeepSeek-V2/V3/R1)
- ssm_decode: Mamba-1 state-space models (a fixed state per sequence)

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
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "int4": 1,
}


def _kv_dtype_bytes(metadata: dict[str, Any], inputs: CalculatorInput) -> int:
    """KV cache element size: the engine's ``kv_cache_dtype`` when set, else
    the model dtype (vLLM's "auto")."""
    kv_dtype = str(inputs.execution_spec_data.get("kv_cache_dtype", "auto"))
    if kv_dtype != "auto":
        return DTYPE_BYTES[kv_dtype]
    return DTYPE_BYTES.get(metadata.get("torch_dtype", "bfloat16"), 2)


def _activation_estimate(metadata: dict[str, Any], inputs: CalculatorInput, tp: int) -> int | None:
    """Peak activation vLLM measures in its startup profiling run.

    Empirical, from the 15 cohort boots (L5, 2026-09-27): the peak tracks the
    output projection's size (vocab x hidden x dtype, per tensor-parallel
    rank) plus two hidden-state buffers per profiled token (the engine's
    ``max_num_batched_tokens``).  It does not track the model's weights: the
    old estimate (10% of weights) was 4x high for Qwen3-32B and 4x low for
    Qwen3-0.6B.  12 of 13 single-GPU points are within ~10%.  Why the output
    projection's size appears in the peak is not traced to a vLLM line; the
    one TP 2 point (Qwen3-8B, measured 0.21 GiB against 0.61) does not fit.
    ``None`` when the config lacks the vocabulary or hidden size.
    """
    vocab = metadata.get("vocab_size")
    hidden = metadata.get("hidden_size")
    if not vocab or not hidden:
        return None
    dtype_bytes = DTYPE_BYTES.get(metadata.get("torch_dtype", "bfloat16"), 2)
    tokens = int(inputs.execution_spec_data.get("max_num_batched_tokens", 2048))
    output_projection = -(-int(vocab) * int(hidden) * dtype_bytes // tp)
    return output_projection + 2 * tokens * int(hidden) * dtype_bytes


def _tensor_parallel(inputs: CalculatorInput) -> int:
    return max(1, int(inputs.execution_spec_data.get("tensor_parallel", 1) or 1))


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

    dtype_bytes = _kv_dtype_bytes(metadata, inputs)
    tp = _tensor_parallel(inputs)

    weight_bytes = metadata.get("total_weight_bytes", 0)
    if weight_bytes <= 0:
        return None
    # Per GPU: each tensor-parallel rank holds 1/tp of the weights and of the
    # KV heads (a KV head is replicated when there are fewer heads than ranks).
    weight_bytes = _per_gpu(int(weight_bytes), tp)
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
        "tensor_parallel": tp,
        "mechanism_detail": "mla",
    }


# ---------------------------------------------------------------------------
# Branch: ssm_decode (Mamba-1 — a fixed-size state per sequence, no KV cache)
# ---------------------------------------------------------------------------


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
    overhead = _common_overhead(weight_bytes, state_cache, activation_estimate, inputs)
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
