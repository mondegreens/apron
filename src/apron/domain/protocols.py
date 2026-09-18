"""Extension-point Protocol contracts (framework-spec.md §1).

Ten Protocols defined here plus ExecutionTarget in primitives.py = 11 total.
Each is a typing.Protocol in the domain layer.  Full conformance suites in
``conformance/`` are a Phase 1a prerequisite; Phase 0 provides these
definitions and one fake-adapter test client per Protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from apron.domain.artifacts import ArtifactSourceObservation
    from apron.domain.schemas.authority import AuthorityContribution
    from apron.domain.schemas.primitives import ArtifactLocator
    from apron.domain.schemas.solutions import DeploymentPlan, PlanningClaim, RenderContext


# ---------------------------------------------------------------------------
# EngineAdapter — resolve, validate, render, verify, classify, extract
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineAdapter(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    def resolve_support(self, model_spec: Any, image_tag: str) -> dict[str, Any]: ...

    def validate(self, plan: DeploymentPlan, target: Any) -> list[str]: ...

    def render(self, plan: DeploymentPlan) -> dict[str, Any]: ...

    def verify(self, plan: DeploymentPlan, target: Any) -> dict[str, Any]: ...

    def classify(self, error: str) -> dict[str, Any]: ...

    def extract(self, error: str, failure_class: str) -> dict[str, int | float | str]: ...

    def extract_schema(self, image_tag: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# ArtifactSourceResolver — resolve an artifact locator to observations
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactSourceResolver(Protocol):
    def resolve(self, locator: ArtifactLocator) -> ArtifactSourceObservation: ...

    def supports(self, source_kind: str) -> bool: ...


# ---------------------------------------------------------------------------
# RenderTarget — render a deployment plan to an external format
# ---------------------------------------------------------------------------


@runtime_checkable
class RenderTarget(Protocol):
    @property
    def target_format(self) -> str: ...

    def render(self, context: RenderContext) -> dict[str, Any]: ...

    def parse(self, data: dict[str, Any]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# EvidenceSource — produce evidence records from execution
# ---------------------------------------------------------------------------


@runtime_checkable
class EvidenceSource(Protocol):
    @property
    def source_name(self) -> str: ...

    def collect(self) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# PlanningSource — produce planning claims
# ---------------------------------------------------------------------------


@runtime_checkable
class PlanningSource(Protocol):
    @property
    def producer_name(self) -> str: ...

    @property
    def producer_version(self) -> str: ...

    def predict(
        self, model_spec: Any, hardware_spec: Any, workload_shape: Any
    ) -> PlanningClaim: ...


# ---------------------------------------------------------------------------
# EvaluationAdapter — run evaluations and produce task attempt records
# ---------------------------------------------------------------------------


@runtime_checkable
class EvaluationAdapter(Protocol):
    @property
    def harness_name(self) -> str: ...

    @property
    def harness_version(self) -> str: ...

    def accepts(self, protocol: dict[str, Any]) -> bool: ...

    def prepare(self, protocol: dict[str, Any]) -> dict[str, Any]: ...

    def execute(self, protocol: dict[str, Any], endpoint: str) -> list[dict[str, Any]]: ...

    def collect(self, attempts: list[dict[str, Any]]) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# RecordStore — persist and retrieve content-addressed records
# ---------------------------------------------------------------------------


@runtime_checkable
class RecordStore(Protocol):
    def store(self, record: dict[str, Any]) -> str: ...

    def retrieve(self, digest: str) -> dict[str, Any] | None: ...

    def search(self, prefix: str) -> list[str]: ...


# ---------------------------------------------------------------------------
# SignalSource — produce orchestration signals
# ---------------------------------------------------------------------------


@runtime_checkable
class SignalSource(Protocol):
    @property
    def signal_type(self) -> str: ...

    def poll(self) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# AuthoritySource — produce authority contributions
# ---------------------------------------------------------------------------


@runtime_checkable
class AuthoritySource(Protocol):
    @property
    def source_type(self) -> str: ...

    @property
    def source_version(self) -> str: ...

    def evaluate(
        self,
        action: str,
        resource: str | None,
        context: dict[str, Any],
    ) -> AuthorityContribution: ...


# ---------------------------------------------------------------------------
# Publisher — publish records to external destinations
# ---------------------------------------------------------------------------


@runtime_checkable
class Publisher(Protocol):
    @property
    def destination(self) -> str: ...

    def publish(self, records: list[dict[str, Any]]) -> dict[str, Any]: ...

    def supports_format(self, format_name: str) -> bool: ...
