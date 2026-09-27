"""F8: attention-free architectures are not ``autoregressive_decode`` (INV-32).

A Mamba-1 config is ``ssm_decode``; any other attention-free model stays an
explicit unknown.
"""

from __future__ import annotations

from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.domain.mechanisms.model_spec_builder import UNKNOWN_MECHANISM, build_model_spec
from apron.domain.schemas.primitives import HardwareSpec

# config.json of state-spaces/mamba-2.8b-hf (fields that matter here, 2026-09-27)
MAMBA_CONFIG = {
    "architectures": ["MambaForCausalLM"],
    "model_type": "mamba",
    "hidden_size": 2560,
    "intermediate_size": 5120,
    "num_hidden_layers": 64,
    "state_size": 16,
    "conv_kernel": 4,
    "vocab_size": 50280,
    "torch_dtype": "float32",
}
# An attention-free model with no Mamba-1 state fields (e.g. RWKV-style).
OTHER_ATTENTION_FREE = {
    "architectures": ["RwkvForCausalLM"],
    "model_type": "rwkv",
    "hidden_size": 2560,
    "num_hidden_layers": 32,
    "torch_dtype": "bfloat16",
}
QWEN_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
}


class _Clock:
    def now(self):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        return datetime(2026, 9, 25, tzinfo=UTC)


def test_mamba_maps_to_ssm_decode() -> None:
    spec = build_model_spec(MAMBA_CONFIG, repository="state-spaces/mamba-2.8b-hf")
    assert [c.mechanism for c in spec.components] == ["ssm_decode"]


def test_other_attention_free_models_stay_unknown() -> None:
    spec = build_model_spec(OTHER_ATTENTION_FREE)
    assert [c.mechanism for c in spec.components] == [UNKNOWN_MECHANISM]


def test_attention_model_still_autoregressive() -> None:
    spec = build_model_spec(QWEN_CONFIG)
    assert [c.mechanism for c in spec.components] == ["autoregressive_decode"]


def test_unknown_attention_free_model_yields_unknown_planning_claim() -> None:
    spec = build_model_spec(OTHER_ATTENTION_FREE)
    metadata = {
        **OTHER_ATTENTION_FREE,
        "components": [c.model_dump(mode="json") for c in spec.components],
    }
    hw = HardwareSpec(gpu_sku="x", total_memory_bytes=24 << 30, compute_capability="8.9")
    claim = CalculatorPlanningSource(clock=_Clock()).predict(metadata, hw, {"isl": 512})
    assert claim.proposed_configuration == {"status": "unknown"}
    assert claim.producer_epistemic_tier == "UNKNOWN"


def test_mamba_state_is_per_sequence_not_per_token() -> None:
    """vLLM v0.29.0 per layer and sequence: conv 5120 x (4 - 1) + SSM
    5120 x 16, in the served dtype (float32 checkpoint served as bfloat16)."""
    spec = build_model_spec(MAMBA_CONFIG)
    metadata = {
        **MAMBA_CONFIG,
        "torch_dtype": "bfloat16",  # plan_pipeline.runtime_dtype
        "total_weight_bytes": 5_535_000_000,
        "components": [c.model_dump(mode="json") for c in spec.components],
    }
    hw = HardwareSpec(gpu_sku="x", total_memory_bytes=24 << 30, compute_capability="8.9")
    source = CalculatorPlanningSource(clock=_Clock())
    short = source.predict(metadata, hw, {"isl": 128, "osl": 128, "max_batch_size": 4})
    long = source.predict(metadata, hw, {"isl": 8192, "osl": 128, "max_batch_size": 4})
    per_sequence = 64 * 5120 * (3 + 16) * 2
    assert short.proposed_configuration["state_per_sequence_bytes"] == per_sequence
    assert short.proposed_configuration["kv_cache_bytes"] == per_sequence * 4
    assert long.proposed_configuration["kv_cache_bytes"] == per_sequence * 4
