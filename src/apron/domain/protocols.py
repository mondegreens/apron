"""Extension-point Protocol contracts (framework-spec.md §1).

Nine Protocols defined here plus ExecutionTarget in primitives.py = 10 total.
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
    from apron.domain.schemas.solutions import DeploymentPlan, PlanningClaim


# ---------------------------------------------------------------------------
# EngineAdapter — boot, profile, and manage an inference engine
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineAdapter(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def engine_version(self) -> str: ...

    def boot(self, plan: DeploymentPlan) -> dict[str, Any]: ...

    def profile_memory(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


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

    def render(self, plan: DeploymentPlan) -> dict[str, Any]: ...

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

    def plan(self, inputs: dict[str, Any]) -> PlanningClaim: ...


# ---------------------------------------------------------------------------
# EvaluationAdapter — run evaluations and produce task attempt records
# ---------------------------------------------------------------------------


@runtime_checkable
class EvaluationAdapter(Protocol):
    @property
    def harness_name(self) -> str: ...

    @property
    def harness_version(self) -> str: ...

    def evaluate(self, protocol: dict[str, Any]) -> list[dict[str, Any]]: ...


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
