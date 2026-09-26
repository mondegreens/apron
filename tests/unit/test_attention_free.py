"""F8: attention-free architectures are not ``autoregressive_decode`` (INV-32)."""

from __future__ import annotations

from apron.adapters.planning.calculator_source import CalculatorPlanningSource
from apron.domain.mechanisms.model_spec_builder import UNKNOWN_MECHANISM, build_model_spec
from apron.domain.schemas.primitives import HardwareSpec

# config.json of state-spaces/mamba-2.8b-hf (fields that matter here)
MAMBA_CONFIG = {
    "architectures": ["MambaForCausalLM"],
    "model_type": "mamba",
    "hidden_size": 2560,
    "num_hidden_layers": 64,
    "state_size": 16,
    "vocab_size": 50280,
    "torch_dtype": "float32",
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


def test_mamba_maps_to_explicit_unknown() -> None:
    spec = build_model_spec(MAMBA_CONFIG, repository="state-spaces/mamba-2.8b-hf")
    assert [c.mechanism for c in spec.components] == [UNKNOWN_MECHANISM]


def test_attention_model_still_autoregressive() -> None:
    spec = build_model_spec(QWEN_CONFIG)
    assert [c.mechanism for c in spec.components] == ["autoregressive_decode"]


def test_mamba_yields_unknown_planning_claim() -> None:
    spec = build_model_spec(MAMBA_CONFIG)
    metadata = {**MAMBA_CONFIG, "components": [c.model_dump(mode="json") for c in spec.components]}
    hw = HardwareSpec(gpu_sku="x", total_memory_bytes=24 << 30, compute_capability="8.9")
    claim = CalculatorPlanningSource(clock=_Clock()).predict(metadata, hw, {"isl": 512})
    assert claim.proposed_configuration == {"status": "unknown"}
    assert claim.producer_epistemic_tier == "UNKNOWN"
