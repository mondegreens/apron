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
    RecordStore,
    RenderTarget,
    SignalSource,
)
from apron.domain.schemas.authority import AuthorityContribution
from apron.domain.schemas.primitives import ArtifactLocator, HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan, PlanningClaim, RenderContext

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

    def resolve_support(self, model_spec: Any, image_tag: str) -> dict[str, Any]:
        return {"supported": True, "tasks": ["generate"]}

    def validate(self, plan: DeploymentPlan, target: Any) -> list[str]:
        return []

    def render(self, plan: DeploymentPlan) -> dict[str, Any]:
        return {"model_id": "test", "tp": plan.tensor_parallel}

    def verify(self, plan: DeploymentPlan, target: Any) -> dict[str, Any]:
        return {
            "model_weight_memory": 8_000_000_000,
            "persistent_consumption": 9_000_000_000,
            "transient_peak_headroom": 1_000_000_000,
            "non_pytorch_increase": 500_000_000,
            "cuda_graph_estimate": 200_000_000,
            "cuda_graph_applied": True,
            "cuda_graph_actual": 180_000_000,
            "available_kv_cache_memory": 6_000_000_000,
            "safety_buffer": 500_000_000,
            "initial_total_memory": 24_000_000_000,
            "initial_free_memory": 23_000_000_000,
            "requested_memory": 16_000_000_000,
            "profiling_shape": {"batch": 1, "seq_len": 128},
            "execution_fingerprint": "fake-fingerprint",
            "target_kind": "local-container",
        }

    def classify(self, error: str) -> dict[str, Any]:
        return {"failure_class": "unknown", "error": error}

    def extract_schema(self, image_tag: str) -> dict[str, Any]:
        return {"architectures": ["LlamaForCausalLM"], "image_tag": image_tag}


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

    def render(self, context: RenderContext) -> dict[str, Any]:
        return {"model_id": context.locator.uri, "tp": context.plan.tensor_parallel}

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

    def predict(self, model_spec: Any, hardware_spec: Any, workload_shape: Any) -> PlanningClaim:
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

    def accepts(self, protocol: dict[str, Any]) -> bool:
        return protocol.get("harness") == "inspect_ai"

    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]:
        return {"prepared": True, "scorer": protocol.get("scorer", "exact_match")}

    def execute(self, protocol: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
        return [
            {"case_id": "case-1", "score": 1.0, "accepted": True},
            {"case_id": "case-2", "score": 0.0, "accepted": False},
        ]

    def collect(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [a["score"] for a in attempts if "score" in a]
        return {
            "aggregate_score": sum(scores) / len(scores) if scores else 0.0,
            "total_attempts": len(attempts),
        }


class FakeRecordStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def store(self, record: dict[str, Any]) -> str:
        from apron.domain.canonical import record_digest_hex

        digest = record_digest_hex(record)
        self._records[digest] = record
        return digest

    def retrieve(self, digest: str) -> dict[str, Any] | None:
        if digest in self._records:
            return self._records[digest]
        matches = [k for k in self._records if k.startswith(digest)]
        if len(matches) == 1:
            return self._records[matches[0]]
        return None

    def search(self, prefix: str) -> list[str]:
        return [k for k in self._records if k.startswith(prefix)]


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
    support = adapter.resolve_support({"arch": "LlamaForCausalLM"}, "vllm:0.8.5")
    assert support["supported"] is True
    plan = DeploymentPlan(tensor_parallel=2)
    assert adapter.validate(plan, None) == []
    rendered = adapter.render(plan)
    assert rendered["tp"] == 2
    verification = adapter.verify(plan, None)
    assert verification["model_weight_memory"] > 0
    assert isinstance(verification, dict)
    classified = adapter.classify("OOM")
    assert "failure_class" in classified
    schema = adapter.extract_schema("vllm:0.8.5")
    assert "architectures" in schema


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
    ctx = RenderContext(
        plan=plan,
        locator=ArtifactLocator(source_kind="huggingface", uri="test/model"),
        hardware=HardwareSpec(
            gpu_sku="RTX 4090",
            total_memory_bytes=25_769_803_776,
            compute_capability="8.9",
        ),
    )
    rendered = renderer.render(ctx)
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
    claim = source.predict({"arch": "Qwen3ForCausalLM"}, {"gpu_sku": "H100"}, {"batch": 32})
    assert claim.producer == "aiconfigurator"
    assert claim.proposed_configuration["batch_size"] == 32


def test_evaluation_adapter_satisfies_protocol():
    adapter = FakeEvaluationAdapter()
    assert isinstance(adapter, EvaluationAdapter)
    proto = {"harness": "inspect_ai", "scorer": "exact_match"}
    assert adapter.accepts(proto) is True
    assert adapter.accepts({"harness": "other"}) is False
    prepared = adapter.prepare(proto)
    assert prepared["prepared"] is True
    results = adapter.execute(proto, "http://localhost:8000")
    assert len(results) == 2
    assert results[0]["accepted"] is True
    assert results[1]["accepted"] is False
    collected = adapter.collect(results)
    assert collected["aggregate_score"] == 0.5
    assert collected["total_attempts"] == 2


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


def test_record_store_satisfies_protocol():
    store = FakeRecordStore()
    assert isinstance(store, RecordStore)
    record = {"type": "test", "value": 42}
    digest = store.store(record)
    assert digest.startswith("1220")
    retrieved = store.retrieve(digest)
    assert retrieved == record
    assert store.retrieve("nonexistent") is None
    results = store.search(digest[:8])
    assert digest in results


def test_publisher_satisfies_protocol():
    pub = FakePublisher()
    assert isinstance(pub, Publisher)
    result = pub.publish([{"digest": "abc"}])
    assert result["published"] == 1
    assert pub.supports_format("recipes_yaml")
    assert not pub.supports_format("unknown")
