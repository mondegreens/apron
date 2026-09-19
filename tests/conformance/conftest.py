"""Shared fake-adapter factories for conformance suites.

A real adapter's test module overrides the fixture by declaring
``pytest_plugins`` and providing its own fixture with the same name.
"""

from __future__ import annotations

from typing import Any

import pytest

from apron.domain.artifacts import ArtifactSourceObservation, FileDigest
from apron.domain.schemas.primitives import ArtifactLocator, HardwareSpec
from apron.domain.schemas.solutions import DeploymentPlan, PlanningClaim, RenderContext

# ---------------------------------------------------------------------------
# EngineAdapter
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
        if plan.tensor_parallel > 1 and plan.tensor_parallel % 2 != 0:
            return [f"TP={plan.tensor_parallel} is not divisible by 2"]
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

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]:
        return {}

    def extract_schema(self, image_tag: str) -> dict[str, Any]:
        return {"architectures": ["LlamaForCausalLM"], "image_tag": image_tag}


@pytest.fixture()
def engine_adapter():
    return FakeEngineAdapter()


# ---------------------------------------------------------------------------
# ArtifactSourceResolver
# ---------------------------------------------------------------------------


class FakeArtifactSourceResolver:
    def resolve(self, locator: ArtifactLocator) -> ArtifactSourceObservation:
        return ArtifactSourceObservation(
            source_kind=locator.source_kind,
            resolved_revision="abc123def456",
            manifest_digest="1220" + "aa" * 32,
            file_digests=(
                FileDigest(path="config.json", sha256="bb" * 32, size_bytes=1024),
                FileDigest(path="model.safetensors.index.json", sha256="cc" * 32, size_bytes=2048),
            ),
        )

    def supports(self, source_kind: str) -> bool:
        return source_kind == "huggingface"


@pytest.fixture()
def artifact_resolver():
    return FakeArtifactSourceResolver()


# ---------------------------------------------------------------------------
# RenderTarget
# ---------------------------------------------------------------------------


class FakeRenderTarget:
    @property
    def target_format(self) -> str:
        return "recipes_yaml"

    def render(self, context: RenderContext) -> dict[str, Any]:
        return {
            "model_id": context.locator.uri,
            "tp": context.plan.tensor_parallel,
            "dtype": context.plan.dtype,
        }

    def parse(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "tensor_parallel": data.get("tp", 1),
            "dtype": data.get("dtype"),
        }


@pytest.fixture()
def render_target():
    return FakeRenderTarget()


# ---------------------------------------------------------------------------
# EvidenceSource
# ---------------------------------------------------------------------------


class FakeEvidenceSource:
    @property
    def source_name(self) -> str:
        return "boot_profiler"

    def collect(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": "test/model",
                "import_status": "owner_attested_boot",
                "type": "memory_profile",
            },
        ]


@pytest.fixture()
def evidence_source():
    return FakeEvidenceSource()


# ---------------------------------------------------------------------------
# PlanningSource
# ---------------------------------------------------------------------------


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
            proposed_configuration={"batch_size": 32, "tp": 1},
            claim_scope="config",
            producer_epistemic_tier="SILICON",
        )


@pytest.fixture()
def planning_source():
    return FakePlanningSource()


# ---------------------------------------------------------------------------
# EvaluationAdapter
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def evaluation_adapter():
    return FakeEvaluationAdapter()


# ---------------------------------------------------------------------------
# ExecutionTarget
# ---------------------------------------------------------------------------


class FakeExecutionTarget:
    def __init__(self) -> None:
        self._provisioned = False
        self._torn_down = False

    @property
    def kind(self) -> str:
        return "local-container"

    @property
    def operator(self) -> str:
        return "apron"

    @property
    def provider(self) -> str | None:
        return "runpod"

    @property
    def hardware(self) -> HardwareSpec:
        return HardwareSpec(
            gpu_sku="H100-SXM-80GB",
            total_memory_bytes=85_899_345_920,
            compute_capability="9.0",
        )

    @property
    def execution_fingerprint(self) -> str:
        return "1220" + "dd" * 32

    def prepare(self) -> None:
        pass

    def provision(self) -> None:
        self._provisioned = True

    def execute(self, command: str) -> Any:
        return {"exit_code": 0, "output": f"ran: {command}"}

    def observe(self) -> dict[str, Any]:
        return {"gpu_utilization": 0.85}

    def collect(self) -> dict[str, Any]:
        return {"logs": [], "metrics": {}}

    def teardown(self) -> None:
        self._torn_down = True


@pytest.fixture()
def execution_target():
    return FakeExecutionTarget()
