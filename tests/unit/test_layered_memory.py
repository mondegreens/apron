"""Per-layer caches counted as vLLM v0.29.0 pages them (domain/mechanisms/layered.py).

Expected bytes are the ones traced from the pinned source
(_dev_notes/cohort-run/hybrid-memory-trace.md, gemma4-memory-trace.md) with the
configs of Qwen/Qwen3.8-27B, Qwen/Qwen3.6-35B-A3B-FP8,
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 and google/gemma-4-31B-it (read
2026-09-27; only the fields the formulas use are copied here).
"""

from __future__ import annotations

from apron.domain.mechanisms.layered import (
    bytes_per_sequence,
    duplicated_weight_bytes,
    family,
    layer_kinds,
)
from apron.domain.mechanisms.model_spec_builder import build_model_spec

L = 262_144

QWEN38 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "full_attention_interval": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 5120,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    },
}
QWEN36 = {
    **QWEN38,
    "text_config": {
        **QWEN38["text_config"],
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "linear_num_value_heads": 32,
    },
}
NEMOTRON = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "hidden_size": 2688,
    "mamba_num_heads": 64,
    "mamba_head_dim": 64,
    "ssm_state_size": 128,
    "n_groups": 8,
    "conv_kernel": 4,
    "mamba_ssm_cache_dtype": "float32",
}
GEMMA4 = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "text_config": {
        "model_type": "gemma4_text",
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_key_value_heads": 16,
        "num_global_key_value_heads": 4,
        "attention_k_eq_v": True,
        "sliding_window": 1024,
        "hidden_size": 5376,
        "num_kv_shared_layers": 0,
    },
}


def _per_sequence(config: dict, in_flight: int = 16_384) -> int | None:
    kinds = layer_kinds(config, tp=1, kv_dtype_bytes=2, model_dtype_bytes=2)
    assert kinds is not None
    return bytes_per_sequence(kinds, L, in_flight_tokens=in_flight)


def test_hybrid_state_models_match_the_traced_reservation() -> None:
    assert _per_sequence(QWEN38) == 17_520_656_384  # 16.32 GiB
    assert _per_sequence(QWEN36) == 5_514_854_400
    assert _per_sequence(NEMOTRON) == 1_725_628_416


def test_gemma4_sliding_and_global_layers_match_the_traced_reservation() -> None:
    # H100: max_num_batched_tokens 8192, two batches in flight
    assert _per_sequence(GEMMA4, in_flight=16_384) == 13_637 * 2_621_440
    # A100: 2048 per batch
    assert _per_sequence(GEMMA4, in_flight=4_096) == 9_797 * 2_621_440
    # vLLM copies k_proj into v_proj for the 10 global layers: 210 MiB of bf16
    assert duplicated_weight_bytes(GEMMA4, tp=1, dtype_bytes=2) == 210 * 2**20


def test_these_models_get_the_layered_mechanism_other_hybrids_stay_unknown() -> None:
    for config in (QWEN38, QWEN36, NEMOTRON, GEMMA4):
        assert family(config) is not None
        assert build_model_spec(config).components[0].mechanism == "layered_decode"
    jamba = {"architectures": ["JambaForCausalLM"], "num_attention_heads": 32}
    spec = build_model_spec(jamba, unmodelled_architectures=frozenset({"JambaForCausalLM"}))
    assert spec.components[0].mechanism == "unknown"


def test_prefix_caching_off_reserves_one_state_page() -> None:
    kinds = layer_kinds(QWEN38, tp=1, kv_dtype_bytes=2, model_dtype_bytes=2)
    assert kinds is not None
    assert bytes_per_sequence(kinds, L, in_flight_tokens=0, prefix_caching=False) == 17_366_515_712
