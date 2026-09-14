"""Tests for the resolution chain and candidate graph."""

from pathlib import Path

from apron.adapters.evidence.hf_hub import FixtureHFHubResolver
from apron.application.orchestration.candidates import CandidateGraph
from apron.application.orchestration.resolution import (
    ResolutionChain,
    StepError,
    StepResult,
)
from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.models import ArtifactSpec
from apron.domain.solutions import DirectEndpoint, InferenceSolution

FIXTURE_DIR = Path("tests/fixtures/external-formats/huggingface-hub")


def _chain() -> ResolutionChain:
    resolver = FixtureHFHubResolver(FIXTURE_DIR)
    return ResolutionChain(resolver=resolver)


# ---------------------------------------------------------------------------
# Full resolution chain
# ---------------------------------------------------------------------------


def test_full_resolution_produces_inference_solution():
    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.ok
    assert isinstance(result.solution, InferenceSolution)
    endpoint = result.solution.endpoints[0]
    assert isinstance(endpoint, DirectEndpoint)
    assert endpoint.deployment_plan_fingerprint is None


def test_resolution_produces_artifact_spec():
    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.artifact_spec is not None
    assert result.artifact_spec.identity.content_digest.startswith("1220")


def test_resolution_identity_is_deterministic():
    r1 = _chain().resolve("Qwen/Qwen3-8B")
    r2 = _chain().resolve("Qwen/Qwen3-8B")
    assert r1.identity is not None
    assert r2.identity is not None
    assert r1.identity.content_digest == r2.identity.content_digest


def test_solution_fingerprint_changes_with_artifact_spec():
    r1 = _chain().resolve("Qwen/Qwen3-8B")
    assert r1.solution is not None
    fp1 = fingerprint_hex(r1.solution)

    r2 = _chain().resolve(
        "Qwen/Qwen3-8B",
        engine_constraints={
            "engine_image_digest": "sha256:abc",
            "resolved_checkpoint_method": "safetensors",
        },
    )
    assert r2.solution is not None
    fp2 = fingerprint_hex(r2.solution)
    assert fp1 != fp2


def test_solution_without_execution_spec():
    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.execution_spec is None
    assert result.ok


def test_solution_with_execution_spec():
    result = _chain().resolve(
        "Qwen/Qwen3-8B",
        engine_constraints={
            "engine_image_digest": "sha256:abc123",
            "resolved_checkpoint_method": "safetensors",
        },
    )
    assert result.execution_spec is not None
    assert result.execution_spec.engine_image_digest == "sha256:abc123"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unsupported_source_returns_step_error():
    result = _chain().resolve("Qwen/Qwen3-8B", source_kind="unknown")
    assert not result.ok
    assert result.error is not None
    assert result.error.step == "resolve_artifact"
    assert "unsupported" in result.error.reason
    assert result.locator is not None


def test_step_result_types():
    ok = StepResult(value=42)
    assert ok.ok
    assert ok.value == 42

    err = StepResult(error=StepError(step="test", reason="fail"))
    assert not err.ok


def test_resolver_exception_produces_step_error():
    """Resolver that throws is caught and wrapped in StepError."""
    from apron.domain.schemas.primitives import ArtifactLocator

    class BrokenResolver:
        def supports(self, source_kind: str) -> bool:
            return True

        def resolve(self, locator: ArtifactLocator):
            raise RuntimeError("model not found")

    chain = ResolutionChain(resolver=BrokenResolver())  # type: ignore[arg-type]
    result = chain.resolve("nonexistent/model")
    assert not result.ok
    assert result.error is not None
    assert result.error.step == "resolve_artifact"
    assert "model not found" in result.error.reason


def test_display_metadata_does_not_change_fingerprint():
    """Changing DISPLAY fields on the solution shouldn't change fingerprint."""
    r = _chain().resolve("Qwen/Qwen3-8B")
    assert r.solution is not None
    fp1 = fingerprint_hex(r.solution)
    fp2 = fingerprint_hex(r.solution)
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# Canonical round-trip (all constructed schema instances)
# ---------------------------------------------------------------------------


def test_artifact_spec_canonical_round_trip():
    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.artifact_spec is not None
    dumped = result.artifact_spec.model_dump(mode="json")
    canonical = canonicalize(dumped)
    restored = ArtifactSpec.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def test_solution_canonical_round_trip():
    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.solution is not None
    dumped = result.solution.model_dump(mode="json")
    canonical = canonicalize(dumped)
    restored = InferenceSolution.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def test_locator_canonical_round_trip():
    from apron.domain.schemas.primitives import ArtifactLocator

    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.locator is not None
    dumped = result.locator.model_dump(mode="json")
    canonical = canonicalize(dumped)
    restored = ArtifactLocator.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def test_observation_canonical_round_trip():
    from apron.domain.artifacts import ArtifactSourceObservation

    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.observation is not None
    dumped = result.observation.model_dump(mode="json")
    canonical = canonicalize(dumped)
    restored = ArtifactSourceObservation.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def test_identity_canonical_round_trip():
    from apron.domain.artifacts.identity import ArtifactIdentity

    result = _chain().resolve("Qwen/Qwen3-8B")
    assert result.identity is not None
    dumped = result.identity.model_dump(mode="json")
    canonical = canonicalize(dumped)
    restored = ArtifactIdentity.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


def test_execution_spec_canonical_round_trip():
    from apron.domain.schemas.models import ExecutionSpec

    result = _chain().resolve(
        "Qwen/Qwen3-8B",
        engine_constraints={
            "engine_image_digest": "sha256:abc",
            "resolved_checkpoint_method": "safetensors",
        },
    )
    assert result.execution_spec is not None
    dumped = result.execution_spec.model_dump(mode="json")
    canonical = canonicalize(dumped)
    restored = ExecutionSpec.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


# ---------------------------------------------------------------------------
# Candidate graph
# ---------------------------------------------------------------------------


def test_candidate_graph_with_resolved():
    result = _chain().resolve("Qwen/Qwen3-8B")
    graph = CandidateGraph.build(resolved=result.artifact_spec)
    assert len(graph.candidates) == 1
    assert graph.candidates[0].source == "resolved"


def test_candidate_graph_with_legacy_entries():
    legacy = [
        {"model_id": "meta-llama/Meta-Llama-3-8B"},
        {"model_id": "Qwen/Qwen3-8B"},
    ]
    graph = CandidateGraph.build(legacy_entries=legacy)
    assert len(graph.candidates) == 2
    assert all(c.source == "legacy" for c in graph.candidates)


def test_legacy_entries_not_deployment_feasible():
    legacy = [{"model_id": "meta-llama/Meta-Llama-3-8B"}]
    graph = CandidateGraph.build(legacy_entries=legacy)
    for c in graph.candidates:
        assert not c.deployment_feasible


def test_resolved_with_execution_spec_is_feasible():
    result = _chain().resolve(
        "Qwen/Qwen3-8B",
        engine_constraints={
            "engine_image_digest": "sha256:abc",
            "resolved_checkpoint_method": "safetensors",
        },
    )
    graph = CandidateGraph.build(
        resolved=result.artifact_spec,
        execution_spec=result.execution_spec,
    )
    assert graph.candidates[0].deployment_feasible


def test_resolved_without_execution_spec_not_feasible():
    result = _chain().resolve("Qwen/Qwen3-8B")
    graph = CandidateGraph.build(resolved=result.artifact_spec)
    assert not graph.candidates[0].deployment_feasible


def test_mixed_graph():
    result = _chain().resolve("Qwen/Qwen3-8B")
    legacy = [{"model_id": "meta-llama/Meta-Llama-3-8B"}]
    graph = CandidateGraph.build(
        resolved=result.artifact_spec,
        legacy_entries=legacy,
    )
    assert len(graph.candidates) == 2
    sources = {c.source for c in graph.candidates}
    assert sources == {"resolved", "legacy"}


def test_empty_graph():
    graph = CandidateGraph.build()
    assert len(graph.candidates) == 0
