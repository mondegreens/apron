"""Layer 0 — Primitives with no cross-schema dependencies.

HardwareSpec, ArtifactLocator, EpistemicStatus (discriminated union),
ClaimScope, and ExecutionTarget Protocol.  CapabilitySignature lives in
``apron.domain.capabilities``.
"""

from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from apron.domain.fingerprints import (
    DISPLAY,
    IDENTITY,
)

# ---------------------------------------------------------------------------
# HardwareSpec
# ---------------------------------------------------------------------------


class HardwareSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    gpu_sku: Annotated[str, IDENTITY]
    total_memory_bytes: Annotated[int, IDENTITY]
    compute_capability: Annotated[str, IDENTITY]
    memory_bandwidth_gbps: Annotated[float | None, IDENTITY] = None
    interconnect: Annotated[str | None, IDENTITY] = None
    topology_position: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# ArtifactLocator
# ---------------------------------------------------------------------------


class ArtifactLocator(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    source_kind: Annotated[str, IDENTITY]
    uri: Annotated[str, IDENTITY]
    requested_revision: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# EpistemicStatus  (discriminated union)
# ---------------------------------------------------------------------------


class DerivedStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Annotated[Literal["derived"], IDENTITY]
    derivation_method: Annotated[str, IDENTITY]
    source_references: Annotated[tuple[str, ...], IDENTITY] = ()


class ProvenConstraintStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Annotated[Literal["proven_constraint"], IDENTITY]
    constraint_source: Annotated[str, IDENTITY]
    verification_method: Annotated[str, IDENTITY]


class PredictedStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Annotated[Literal["predicted"], IDENTITY]
    prediction_method: Annotated[str, IDENTITY]
    uncertainty_lower: Annotated[float | None, IDENTITY] = None
    uncertainty_upper: Annotated[float | None, IDENTITY] = None
    calibration_scope: Annotated[str | None, IDENTITY] = None


class MeasuredStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Annotated[Literal["measured"], IDENTITY]
    measurement_method: Annotated[str, IDENTITY]
    execution_fingerprint: Annotated[str, IDENTITY]


EpistemicStatus = Annotated[
    DerivedStatus | ProvenConstraintStatus | PredictedStatus | MeasuredStatus,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# ClaimScope
# ---------------------------------------------------------------------------

ClaimScope = Literal[
    "config",
    "boot",
    "memory",
    "remediation",
    "serving_performance",
    "task_outcome",
    "outcome_economics",
]


# ---------------------------------------------------------------------------
# ExecutionTarget Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionTarget(Protocol):
    """Extension-point contract for execution environments (ADR-001 §4).

    Kinds: local-container, remote-container, rented-provider.
    Every evidence record carries the target's identity fields.
    """

    @property
    def kind(self) -> str: ...

    @property
    def operator(self) -> str: ...

    @property
    def provider(self) -> str | None: ...

    @property
    def hardware(self) -> HardwareSpec: ...

    @property
    def execution_fingerprint(self) -> str: ...

    def prepare(self) -> None: ...

    def provision(self) -> None: ...

    def execute(self, command: str) -> Any: ...

    def observe(self) -> dict[str, Any]: ...

    def collect(self) -> dict[str, Any]: ...

    def teardown(self) -> None: ...
