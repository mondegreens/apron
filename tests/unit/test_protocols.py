"""Step 10 tests: 9 extension-point Protocols — each with a fake-adapter test client."""

from __future__ import annotations

from typing import Any

from apron.domain.artifacts import (
    ArtifactSourceObservation,
    FileDigest,
)
from apron.domain.protocols import (
    ArtifactSourceResolver,
    AuthoritySource,
    EngineAdapter,
    EvaluationAdapter,
    EvidenceSource,
    PlanningSource,
    Publisher,
    RenderTarget,
    SignalSource,
)
from apron.domain.schemas.authority import AuthorityContribution
from apron.domain.schemas.primitives import ArtifactLocator
from apron.domain.schemas.solutions import DeploymentPlan, PlanningClaim

# ---------------------------------------------------------------------------
# Fake adapters
# ---------------------------------------------------------------------------


class FakeEngineAdapter:
    @property
    def engine_name(self) -> str:
        return "vllm"

    @property
    def engine_version(self) -> str:
        return "0.8.5"

    def boot(self, plan: DeploymentPlan) -> dict[str, Any]:
        return {"status": "booted", "tp": plan.tensor_parallel}

    def profile_memory(self) -> dict[str, Any]:
        return {"model_weight_memory": 8_000_000_000}

    def shutdown(self) -> None:
        pass


class FakeArtifactSourceResolver:
    def resolve(self, locator: ArtifactLocator) -> ArtifactSourceObservation:
        return ArtifactSourceObservation(
            source_kind=locator.source_kind,
            resolved_revision="abc123",
            file_digests=(FileDigest(path="config.json", sha256="aa" * 32, size_bytes=1024),),
        )

    def supports(self, source_kind: str) -> bool:
        return source_kind == "huggingface"


class FakeRenderTarget:
    @property
    def target_format(self) -> str:
        return "recipes_yaml"

    def render(self, plan: DeploymentPlan) -> dict[str, Any]:
        return {"model_id": "test", "tp": plan.tensor_parallel}

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"tensor_parallel": data.get("tp", 1)}


class FakeEvidenceSource:
    @property
    def source_name(self) -> str:
        return "boot_profiler"

    def collect(self) -> list[dict[str, Any]]:
        return [{"type": "memory_profile", "total": 24_000_000_000}]


class FakePlanningSource:
    @property
    def producer_name(self) -> str:
        return "aiconfigurator"

    @property
    def producer_version(self) -> str:
        return "1.0.0"

    def plan(self, inputs: dict[str, Any]) -> PlanningClaim:
        return PlanningClaim(
            producer=self.producer_name,
            version=self.producer_version,
            input_fingerprint="1220" + "ab" * 32,
            proposed_configuration={"batch_size": 32},
            claim_scope="config",
            producer_epistemic_tier="SILICON",
        )


class FakeEvaluationAdapter:
    @property
    def harness_name(self) -> str:
        return "inspect_ai"

    @property
    def harness_version(self) -> str:
        return "0.4.0"

    def evaluate(self, protocol: dict[str, Any]) -> list[dict[str, Any]]:
        return [{"case_id": "case-1", "score": 1.0, "accepted": True}]


class FakeSignalSource:
    @property
    def signal_type(self) -> str:
        return "cost_threshold"

    def poll(self) -> list[dict[str, Any]]:
        return []


class FakeAuthoritySource:
    @property
    def source_type(self) -> str:
        return "owner_policy"

    @property
    def source_version(self) -> str:
        return "1.0"

    def evaluate(
        self,
        action: str,
        resource: str | None,
        context: dict[str, Any],
    ) -> AuthorityContribution:
        return AuthorityContribution(
            source_type=self.source_type,
            source_version=self.source_version,
            principal="owner",
            decision="permit",
            action=action,
            resource=resource,
        )


class FakePublisher:
    @property
    def destination(self) -> str:
        return "recipes_repo"

    def publish(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {"published": len(records)}

    def supports_format(self, format_name: str) -> bool:
        return format_name == "recipes_yaml"


# ---------------------------------------------------------------------------
# Protocol satisfaction tests
# ---------------------------------------------------------------------------


def test_engine_adapter_satisfies_protocol():
    adapter = FakeEngineAdapter()
    assert isinstance(adapter, EngineAdapter)
    result = adapter.boot(DeploymentPlan(tensor_parallel=2))
    assert result["tp"] == 2
    assert adapter.profile_memory()["model_weight_memory"] > 0
    adapter.shutdown()


def test_artifact_source_resolver_satisfies_protocol():
    resolver = FakeArtifactSourceResolver()
    assert isinstance(resolver, ArtifactSourceResolver)
    loc = ArtifactLocator(source_kind="huggingface", uri="Qwen/Qwen3-8B")
    obs = resolver.resolve(loc)
    assert obs.source_kind == "huggingface"
    assert resolver.supports("huggingface")
    assert not resolver.supports("oci")


def test_render_target_satisfies_protocol():
    renderer = FakeRenderTarget()
    assert isinstance(renderer, RenderTarget)
    plan = DeploymentPlan(tensor_parallel=4)
    rendered = renderer.render(plan)
    assert rendered["tp"] == 4
    parsed = renderer.parse(rendered)
    assert parsed["tensor_parallel"] == 4


def test_evidence_source_satisfies_protocol():
    source = FakeEvidenceSource()
    assert isinstance(source, EvidenceSource)
    records = source.collect()
    assert len(records) == 1


def test_planning_source_satisfies_protocol():
    source = FakePlanningSource()
    assert isinstance(source, PlanningSource)
    claim = source.plan({"model": "Qwen3-8B"})
    assert claim.producer == "aiconfigurator"
    assert claim.proposed_configuration["batch_size"] == 32


def test_evaluation_adapter_satisfies_protocol():
    adapter = FakeEvaluationAdapter()
    assert isinstance(adapter, EvaluationAdapter)
    results = adapter.evaluate({"scorer": "exact_match"})
    assert len(results) == 1
    assert results[0]["accepted"] is True


def test_signal_source_satisfies_protocol():
    source = FakeSignalSource()
    assert isinstance(source, SignalSource)
    signals = source.poll()
    assert signals == []


def test_authority_source_satisfies_protocol():
    source = FakeAuthoritySource()
    assert isinstance(source, AuthoritySource)
    contribution = source.evaluate("submit", None, {})
    assert contribution.decision == "permit"
    assert contribution.source_type == "owner_policy"


def test_publisher_satisfies_protocol():
    pub = FakePublisher()
    assert isinstance(pub, Publisher)
    result = pub.publish([{"digest": "abc"}])
    assert result["published"] == 1
    assert pub.supports_format("recipes_yaml")
    assert not pub.supports_format("unknown")
