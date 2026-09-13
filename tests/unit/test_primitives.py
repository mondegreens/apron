"""Step 2 tests: Layer 0 primitives — round-trip, classification, ExecutionTarget fake adapter."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from apron.domain.canonical import canonicalize
from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import assert_fully_classified
from apron.domain.schemas.primitives import (
    ArtifactLocator,
    ClaimScope,
    DerivedStatus,
    EpistemicStatus,
    ExecutionTarget,
    HardwareSpec,
    MeasuredStatus,
    PredictedStatus,
    ProvenConstraintStatus,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_HARDWARE = HardwareSpec(
    gpu_sku="NVIDIA RTX 4090",
    total_memory_bytes=25_769_803_776,
    compute_capability="8.9",
    memory_bandwidth_gbps=1008.0,
    interconnect="PCIe",
)


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_hardware_spec_fully_classified():
    assert_fully_classified(HardwareSpec)


def test_artifact_locator_fully_classified():
    assert_fully_classified(ArtifactLocator)


def test_capability_signature_fully_classified():
    assert_fully_classified(CapabilitySignature)


def test_epistemic_status_variants_fully_classified():
    assert_fully_classified(DerivedStatus)
    assert_fully_classified(ProvenConstraintStatus)
    assert_fully_classified(PredictedStatus)
    assert_fully_classified(MeasuredStatus)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_hardware_spec_round_trip():
    _round_trip(HardwareSpec, _HARDWARE)


def test_artifact_locator_round_trip():
    loc = ArtifactLocator(
        source_kind="huggingface",
        uri="Qwen/Qwen3-8B",
        requested_revision="main",
    )
    _round_trip(ArtifactLocator, loc)


def test_capability_signature_round_trip():
    cap = CapabilitySignature(
        operation="text_generation",
        required_inputs=("text",),
        output_representation="generated_text",
    )
    _round_trip(CapabilitySignature, cap)


def test_capability_text_image_combination():
    cap = CapabilitySignature(
        operation="text_generation",
        required_inputs=("text", "image"),
        output_representation="generated_text",
    )
    restored = _round_trip(CapabilitySignature, cap)
    assert restored.required_inputs == ("text", "image")


def test_epistemic_derived_round_trip():
    s = DerivedStatus(kind="derived", derivation_method="formula_v1")
    _round_trip(DerivedStatus, s)


def test_epistemic_proven_constraint_round_trip():
    s = ProvenConstraintStatus(
        kind="proven_constraint",
        constraint_source="vllm/kv_cache_interface.py",
        verification_method="source_code_analysis",
    )
    _round_trip(ProvenConstraintStatus, s)


def test_epistemic_predicted_round_trip():
    s = PredictedStatus(
        kind="predicted",
        prediction_method="linear_extrapolation",
        uncertainty_lower=0.85,
        uncertainty_upper=1.15,
        calibration_scope="batch_size_32_to_64",
    )
    _round_trip(PredictedStatus, s)


def test_epistemic_measured_round_trip():
    s = MeasuredStatus(
        kind="measured",
        measurement_method="boot_profiling",
        execution_fingerprint="1220" + "ab" * 32,
    )
    _round_trip(MeasuredStatus, s)


def test_epistemic_discriminated_union():
    adapter = TypeAdapter(EpistemicStatus)

    measured_json = '{"kind":"measured","measurement_method":"boot","execution_fingerprint":"1220' + "ab" * 32 + '"}'
    parsed = adapter.validate_json(measured_json)
    assert isinstance(parsed, MeasuredStatus)

    predicted_json = '{"kind":"predicted","prediction_method":"formula"}'
    parsed = adapter.validate_json(predicted_json)
    assert isinstance(parsed, PredictedStatus)


def test_claim_scope_values():
    adapter = TypeAdapter(ClaimScope)
    for scope in ("config", "boot", "memory", "remediation", "serving_performance", "task_outcome", "outcome_economics"):
        assert adapter.validate_python(scope) == scope


# ---------------------------------------------------------------------------
# ExecutionTarget fake adapter
# ---------------------------------------------------------------------------


class FakeExecutionTarget:
    """Fake adapter proving the Protocol is implementable."""

    def __init__(self, hw: HardwareSpec) -> None:
        self._hw = hw

    @property
    def kind(self) -> str:
        return "local-container"

    @property
    def operator(self) -> str:
        return "self"

    @property
    def provider(self) -> str | None:
        return None

    @property
    def hardware(self) -> HardwareSpec:
        return self._hw

    @property
    def execution_fingerprint(self) -> str:
        return "1220" + "00" * 32

    def prepare(self) -> None:
        pass

    def provision(self) -> None:
        pass

    def execute(self, command: str) -> Any:
        return {"exit_code": 0}

    def observe(self) -> dict[str, Any]:
        return {"status": "running"}

    def collect(self) -> dict[str, Any]:
        return {"logs": []}

    def teardown(self) -> None:
        pass


def test_fake_adapter_satisfies_protocol():
    target = FakeExecutionTarget(_HARDWARE)
    assert isinstance(target, ExecutionTarget)


def test_fake_adapter_properties():
    target = FakeExecutionTarget(_HARDWARE)
    assert target.kind == "local-container"
    assert target.operator == "self"
    assert target.provider is None
    assert target.hardware == _HARDWARE
    assert target.execution_fingerprint.startswith("1220")


def test_fake_adapter_lifecycle():
    target = FakeExecutionTarget(_HARDWARE)
    target.prepare()
    target.provision()
    result = target.execute("vllm serve")
    assert result["exit_code"] == 0
    obs = target.observe()
    assert "status" in obs
    collected = target.collect()
    assert "logs" in collected
    target.teardown()
