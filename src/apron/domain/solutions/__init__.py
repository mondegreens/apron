"""Layer 4 — EndpointBinding (ADR-013) and InferenceSolution (ADR-011 §4).

EndpointBinding is a discriminated union of topology types.
InferenceSolution holds endpoint bindings with routing policy and role
bindings.  Every endpoint retains fingerprints of its own ModelSpec,
ArtifactSpec, ExecutionSpec, CapabilitySignatures, provider/target
identity, and DeploymentPlan.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex

# ---------------------------------------------------------------------------
# EndpointBinding (ADR-013 §1, discriminated union)
# ---------------------------------------------------------------------------


class DirectEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    binding: Annotated[Literal["direct_endpoint"], IDENTITY]
    model_spec_fingerprint: Annotated[FingerprintHex, IDENTITY]
    artifact_spec_fingerprint: Annotated[FingerprintHex | None, IDENTITY] = None
    execution_spec_fingerprint: Annotated[FingerprintHex | None, IDENTITY] = None
    capabilities: Annotated[tuple[FingerprintHex, ...], IDENTITY] = ()
    provider: Annotated[str | None, IDENTITY] = None
    target_kind: Annotated[str | None, IDENTITY] = None
    deployment_plan_fingerprint: Annotated[FingerprintHex | None, IDENTITY] = None
    provider_opaque: Annotated[bool, IDENTITY] = False


class LogicalRoute(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    binding: Annotated[Literal["logical_route"], IDENTITY]
    endpoints: Annotated[tuple[FingerprintHex, ...], IDENTITY]
    routing_policy: Annotated[str, IDENTITY]
    routing_opaque: Annotated[bool, IDENTITY] = False


class ReplicaPool(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    binding: Annotated[Literal["replica_pool"], IDENTITY]
    replica_fingerprints: Annotated[tuple[FingerprintHex, ...], IDENTITY]


class DistributedExecutionGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    binding: Annotated[Literal["distributed_execution_group"], IDENTITY]
    worker_fingerprints: Annotated[tuple[FingerprintHex, ...], IDENTITY]
    worker_roles: Annotated[tuple[str, ...], IDENTITY] = ()
    model_placement: Annotated[str | None, IDENTITY] = None
    interconnect: Annotated[str | None, IDENTITY] = None


EndpointBinding = Annotated[
    DirectEndpoint | LogicalRoute | ReplicaPool | DistributedExecutionGroup,
    Field(discriminator="binding"),
]


# ---------------------------------------------------------------------------
# InferenceSolution (ADR-011 §4, ADR-013)
# ---------------------------------------------------------------------------


class RoleBinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    role: Annotated[str, IDENTITY]
    endpoint_fingerprint: Annotated[FingerprintHex, IDENTITY]


class InferenceSolution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    endpoints: Annotated[
        tuple[DirectEndpoint | LogicalRoute | ReplicaPool | DistributedExecutionGroup, ...],
        IDENTITY,
    ]
    role_bindings: Annotated[tuple[RoleBinding, ...], IDENTITY] = ()
    routing_policy: Annotated[str | None, IDENTITY] = None
    fallback_policy: Annotated[str | None, IDENTITY] = None
    escalation_policy: Annotated[str | None, IDENTITY] = None
