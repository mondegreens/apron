"""Layer 2 — Model, execution, and evidence schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from apron.domain.artifacts import (
    ArtifactSourceObservation,
    ModelLineage,
)
from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex  # noqa: TC001
from apron.domain.mechanisms import ComponentMechanism
from apron.domain.schemas.primitives import ArtifactLocator


# ---------------------------------------------------------------------------
# ArtifactSpec (ADR-006)
# ---------------------------------------------------------------------------


class ArtifactSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    identity: Annotated[ArtifactIdentity, IDENTITY]
    locators: Annotated[tuple[ArtifactLocator, ...], DISPLAY] = ()
    observations: Annotated[tuple[ArtifactSourceObservation, ...], DISPLAY] = ()
    weight_manifest_digest: Annotated[str | None, IDENTITY] = None
    index_manifest_digest: Annotated[str | None, IDENTITY] = None
    config_digest: Annotated[str | None, IDENTITY] = None
    quantization_config_digest: Annotated[str | None, IDENTITY] = None
    tokenizer_identity: Annotated[str | None, IDENTITY] = None
    chat_template_identity: Annotated[str | None, IDENTITY] = None
    attributed_claims: Annotated[tuple[str, ...], DISPLAY] = ()
    claimed_lineage: Annotated[ModelLineage | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# ArtifactRelation (ADR-006, discriminated union)
# ---------------------------------------------------------------------------


class PublishedByOwner(BaseModel):
    model_config = ConfigDict(frozen=True)
    relation: Annotated[Literal["published_by_owner"], IDENTITY]
    publisher: Annotated[str, IDENTITY]
    artifact_digest: Annotated[str, IDENTITY]


class ClaimedDerivedFrom(BaseModel):
    model_config = ConfigDict(frozen=True)
    relation: Annotated[Literal["claimed_derived_from"], IDENTITY]
    source_artifact_digest: Annotated[str, IDENTITY]
    claim_basis: Annotated[str, IDENTITY]


class ReproduciblyDerivedFrom(BaseModel):
    model_config = ConfigDict(frozen=True)
    relation: Annotated[Literal["reproducibly_derived_from"], IDENTITY]
    source_artifact_digest: Annotated[str, IDENTITY]
    transform_spec_digest: Annotated[str, IDENTITY]


class StructurallyCompatibleWith(BaseModel):
    model_config = ConfigDict(frozen=True)
    relation: Annotated[Literal["structurally_compatible_with"], IDENTITY]
    other_artifact_digest: Annotated[str, IDENTITY]
    matching_criteria: Annotated[str, IDENTITY]


class QualityComparedWith(BaseModel):
    model_config = ConfigDict(frozen=True)
    relation: Annotated[Literal["quality_compared_with"], IDENTITY]
    other_artifact_digest: Annotated[str, IDENTITY]
    task_protocol_digest: Annotated[str, IDENTITY]


class TokenizerCompatibleWith(BaseModel):
    model_config = ConfigDict(frozen=True)
    relation: Annotated[Literal["tokenizer_compatible_with"], IDENTITY]
    other_artifact_digest: Annotated[str, IDENTITY]
    tokenizer_hash: Annotated[str, IDENTITY]


ArtifactRelation = Annotated[
    PublishedByOwner
    | ClaimedDerivedFrom
    | ReproduciblyDerivedFrom
    | StructurallyCompatibleWith
    | QualityComparedWith
    | TokenizerCompatibleWith,
    Field(discriminator="relation"),
]


# ---------------------------------------------------------------------------
# ExecutionSpec (ADR-006)
# ---------------------------------------------------------------------------


class ExecutionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    engine_image_digest: Annotated[str, IDENTITY]
    resolved_checkpoint_method: Annotated[str, IDENTITY]
    implementation_overrides: Annotated[dict[str, str], IDENTITY] = {}
    selected_kernels: Annotated[tuple[str, ...], IDENTITY] = ()


# ---------------------------------------------------------------------------
# ModelSpec (ADR-007 §4, ADR-011 §4)
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    repository: Annotated[str | None, IDENTITY] = None
    immutable_revision: Annotated[str | None, IDENTITY] = None
    license: Annotated[str | None, DISPLAY] = None
    files: Annotated[tuple[str, ...], IDENTITY] = ()
    component_bytes_dtype: Annotated[dict[str, str], IDENTITY] = {}
    remote_code_required: Annotated[bool, IDENTITY] = False
    lineage: Annotated[ModelLineage | None, DISPLAY] = None
    publisher_claims: Annotated[tuple[str, ...], DISPLAY] = ()
    components: Annotated[tuple[ComponentMechanism, ...], IDENTITY]
    edges: Annotated[tuple[tuple[str, str], ...], IDENTITY] = ()


# ---------------------------------------------------------------------------
# QualityEvidence (ADR-006, ADR-011)
# ---------------------------------------------------------------------------


class QualityEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    task_attempt_fingerprints: Annotated[tuple[FingerprintHex, ...], IDENTITY]
    evaluation_protocol_fingerprint: Annotated[FingerprintHex, IDENTITY]
    aggregate_score: Annotated[float | None, DISPLAY] = None
    score_breakdown: Annotated[dict[str, float] | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# CompatibilityEvidence (ADR-006)
# ---------------------------------------------------------------------------


class CompatibilityEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    candidate_fingerprint: Annotated[FingerprintHex, IDENTITY]
    execution_fingerprint: Annotated[str, IDENTITY]
    compatible: Annotated[bool, IDENTITY]
    details: Annotated[str | None, DISPLAY] = None
