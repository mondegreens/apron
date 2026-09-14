"""Step 4 tests: Layer 2 — models, calculator, evidence, artifact relations."""

from __future__ import annotations

from typing import Any

from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import assert_fully_classified
from apron.domain.mechanisms import (
    AudioWorkload,
    CalculatorInput,
    ComponentMechanism,
    ImageWorkload,
    TextWorkload,
    WorkloadShape,
    calculate,
    clear_calculator_registry,
    register_calculator,
)
from apron.domain.schemas.models import (
    ArtifactRelation,
    ArtifactSpec,
    ClaimedDerivedFrom,
    CompatibilityEvidence,
    ExecutionSpec,
    ModelSpec,
    PublishedByOwner,
    QualityComparedWith,
    QualityEvidence,
    ReproduciblyDerivedFrom,
    StructurallyCompatibleWith,
    TokenizerCompatibleWith,
)
from apron.domain.schemas.primitives import ArtifactLocator, HardwareSpec
from apron.domain.schemas.solutions import PlanningClaim
from pydantic import TypeAdapter


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


_HW = HardwareSpec(
    gpu_sku="NVIDIA RTX 4090",
    total_memory_bytes=25_769_803_776,
    compute_capability="8.9",
)

_FP = "1220" + "ab" * 32


def setup_function() -> None:
    clear_calculator_registry()


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_component_mechanism_classified():
    assert_fully_classified(ComponentMechanism)


def test_workload_shape_variants_classified():
    assert_fully_classified(TextWorkload)
    assert_fully_classified(AudioWorkload)
    assert_fully_classified(ImageWorkload)


def test_artifact_spec_classified():
    assert_fully_classified(ArtifactSpec)


def test_execution_spec_classified():
    assert_fully_classified(ExecutionSpec)


def test_model_spec_classified():
    assert_fully_classified(ModelSpec)


def test_quality_evidence_classified():
    assert_fully_classified(QualityEvidence)


def test_compatibility_evidence_classified():
    assert_fully_classified(CompatibilityEvidence)


def test_planning_claim_classified():
    assert_fully_classified(PlanningClaim)


def test_artifact_relation_variants_classified():
    assert_fully_classified(PublishedByOwner)
    assert_fully_classified(ClaimedDerivedFrom)
    assert_fully_classified(ReproduciblyDerivedFrom)
    assert_fully_classified(StructurallyCompatibleWith)
    assert_fully_classified(QualityComparedWith)
    assert_fully_classified(TokenizerCompatibleWith)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_component_mechanism_round_trip():
    m = ComponentMechanism(mechanism="autoregressive_decode", role="decoder")
    _round_trip(ComponentMechanism, m)


def test_workload_shape_round_trip():
    adapter = TypeAdapter(WorkloadShape)
    text = TextWorkload(kind="text", input_length=512, output_length=128)
    canonical = canonicalize(text.model_dump(mode="json"))
    restored = adapter.validate_json(canonical)
    assert isinstance(restored, TextWorkload)
    assert restored.input_length == 512


def test_artifact_spec_round_trip():
    identity = ArtifactIdentity(content_digest=_FP)
    spec = ArtifactSpec(
        identity=identity,
        locators=(ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B"),),
        config_digest="aabb" * 16,
    )
    _round_trip(ArtifactSpec, spec)


def test_execution_spec_round_trip():
    es = ExecutionSpec(
        engine_image_digest="1220" + "cc" * 32,
        resolved_checkpoint_method="auto",
    )
    _round_trip(ExecutionSpec, es)


def test_model_spec_round_trip():
    ms = ModelSpec(
        repository="Qwen/Qwen3-8B",
        immutable_revision="abc123",
        components=(ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),),
    )
    _round_trip(ModelSpec, ms)


def test_model_spec_component_graph():
    ms = ModelSpec(
        components=(
            ComponentMechanism(mechanism="media_encoder", role="vision_encoder"),
            ComponentMechanism(mechanism="projector", role="vision_projector"),
            ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),
        ),
        edges=(("vision_encoder", "vision_projector"), ("vision_projector", "decoder")),
    )
    assert len(ms.components) == 3
    assert len(ms.edges) == 2
    _round_trip(ModelSpec, ms)


def test_quality_evidence_round_trip():
    qe = QualityEvidence(
        task_attempt_fingerprints=(_FP,),
        evaluation_protocol_fingerprint=_FP,
        aggregate_score=0.95,
    )
    _round_trip(QualityEvidence, qe)


def test_compatibility_evidence_round_trip():
    ce = CompatibilityEvidence(
        candidate_fingerprint=_FP,
        execution_fingerprint=_FP,
        compatible=True,
    )
    _round_trip(CompatibilityEvidence, ce)


def test_planning_claim_round_trip():
    pc = PlanningClaim(
        producer="aiconfigurator",
        version="1.0.0",
        input_fingerprint=_FP,
        proposed_configuration={"batch_size": 32, "tp": 1},
        claim_scope="config",
        producer_epistemic_tier="SILICON",
    )
    _round_trip(PlanningClaim, pc)


# ---------------------------------------------------------------------------
# ArtifactRelation discriminated union
# ---------------------------------------------------------------------------


def test_artifact_relation_discriminated_union():
    adapter = TypeAdapter(ArtifactRelation)

    pub = PublishedByOwner(
        relation="published_by_owner",
        publisher="Qwen",
        artifact_digest=_FP,
    )
    canonical = canonicalize(pub.model_dump(mode="json"))
    parsed = adapter.validate_json(canonical)
    assert isinstance(parsed, PublishedByOwner)

    derived = ClaimedDerivedFrom(
        relation="claimed_derived_from",
        source_artifact_digest=_FP,
        claim_basis="name_similarity",
    )
    canonical = canonicalize(derived.model_dump(mode="json"))
    parsed = adapter.validate_json(canonical)
    assert isinstance(parsed, ClaimedDerivedFrom)


# ---------------------------------------------------------------------------
# calculator dispatch
# ---------------------------------------------------------------------------


def _make_calculator_input(mechanism: str = "autoregressive_decode") -> CalculatorInput:
    return CalculatorInput(
        mechanism=ComponentMechanism(mechanism=mechanism, role="decoder"),
        workload=TextWorkload(kind="text", input_length=512, output_length=128),
        artifact_metadata={"total_params": 8_000_000_000},
        hardware=_HW,
        execution_spec_data={"engine": "vllm", "version": "0.8.5"},
    )


def test_unimplemented_mechanism_returns_none():
    result = calculate(_make_calculator_input("unknown_mechanism"))
    assert result is None


def test_registered_calculator_dispatches():
    @register_calculator("autoregressive_decode")
    def _calc(inputs: CalculatorInput) -> dict[str, Any] | None:
        return {
            "batch_size": 32,
            "mechanism": inputs.mechanism.mechanism,
        }

    result = calculate(_make_calculator_input("autoregressive_decode"))
    assert result is not None
    assert result["batch_size"] == 32
    assert result["mechanism"] == "autoregressive_decode"


def test_different_mechanisms_dispatch_independently():
    @register_calculator("autoregressive_decode")
    def _ar(inputs: CalculatorInput) -> dict[str, Any] | None:
        return {"type": "ar"}

    @register_calculator("single_pass_pooling")
    def _pool(inputs: CalculatorInput) -> dict[str, Any] | None:
        return {"type": "pool"}

    ar_result = calculate(_make_calculator_input("autoregressive_decode"))
    assert ar_result is not None
    assert ar_result["type"] == "ar"
    pool_result = calculate(_make_calculator_input("single_pass_pooling"))
    assert pool_result is not None
    assert pool_result["type"] == "pool"
    assert calculate(_make_calculator_input("latent_denoising")) is None


def test_calculator_duplicate_registration_raises():
    @register_calculator("test_mech")
    def _first(inputs: CalculatorInput) -> dict[str, Any] | None:
        return None

    try:

        @register_calculator("test_mech")
        def _second(inputs: CalculatorInput) -> dict[str, Any] | None:
            return None

        raise RuntimeError("should have raised")
    except ValueError as e:
        assert "already registered" in str(e)


def test_calculator_consumes_all_inputs():
    @register_calculator("autoregressive_decode")
    def _calc(inputs: CalculatorInput) -> dict[str, Any] | None:
        assert inputs.artifact_metadata["total_params"] == 8_000_000_000
        assert inputs.hardware.gpu_sku == "NVIDIA RTX 4090"
        assert inputs.execution_spec_data["engine"] == "vllm"
        assert isinstance(inputs.workload, TextWorkload)
        return {"verified": True}

    result = calculate(_make_calculator_input())
    assert result is not None
    assert result["verified"] is True
