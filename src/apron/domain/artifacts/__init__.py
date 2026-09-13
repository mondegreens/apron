"""Layer 1 — Artifact source observations, lineage, and transform specs.

ArtifactIdentity lives in ``apron.domain.artifacts.identity``.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import (
    DISPLAY,
    IDENTITY,
)

# ---------------------------------------------------------------------------
# FileDigest — per-file content digest within an observation
# ---------------------------------------------------------------------------


class FileDigest(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Annotated[str, IDENTITY]
    sha256: Annotated[str, IDENTITY]
    size_bytes: Annotated[int, IDENTITY]


# ---------------------------------------------------------------------------
# ArtifactSourceObservation (ADR-002 §8)
# ---------------------------------------------------------------------------


class ArtifactSourceObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    source_kind: Annotated[str, IDENTITY]
    resolved_revision: Annotated[str, IDENTITY]
    manifest_digest: Annotated[str | None, IDENTITY] = None
    file_digests: Annotated[tuple[FileDigest, ...], IDENTITY]
    license_observed: Annotated[str | None, DISPLAY] = None
    gating_observed: Annotated[str | None, DISPLAY] = None
    publisher_metadata: Annotated[dict[str, str] | None, DISPLAY] = None
    observed_at: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# ModelLineage (ADR-006)
# ---------------------------------------------------------------------------


class ModelLineage(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    family: Annotated[str, IDENTITY]
    publishers: Annotated[tuple[str, ...], IDENTITY] = ()
    derivatives: Annotated[tuple[str, ...], DISPLAY] = ()


# ---------------------------------------------------------------------------
# QuantizationSpec (ADR-006)
# ---------------------------------------------------------------------------


class QuantizationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    weight_format: Annotated[str, IDENTITY]
    activation_format: Annotated[str | None, IDENTITY] = None
    kv_cache_format: Annotated[str | None, IDENTITY] = None
    scheme: Annotated[str, IDENTITY]
    bits: Annotated[int | None, IDENTITY] = None
    group_size: Annotated[int | None, IDENTITY] = None
    block_shape: Annotated[tuple[int, ...] | None, IDENTITY] = None
    scale_representation: Annotated[str | None, IDENTITY] = None
    zero_point_representation: Annotated[str | None, IDENTITY] = None
    excluded_modules: Annotated[tuple[str, ...], IDENTITY] = ()
    calibration_recipe: Annotated[str | None, IDENTITY] = None
    calibration_dataset: Annotated[str | None, IDENTITY] = None
    producer_version: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# OfflineTransformSpec (ADR-006)
# ---------------------------------------------------------------------------


class OfflineTransformSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    producer: Annotated[str, IDENTITY]
    version: Annotated[str, IDENTITY]
    recipe: Annotated[str | None, IDENTITY] = None
    calibration_data_identity: Annotated[str | None, IDENTITY] = None
    output_artifact_digest: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# RuntimeTransformSpec (ADR-006)
# ---------------------------------------------------------------------------


class RuntimeTransformSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    kind: Annotated[str, IDENTITY]
    engine_version: Annotated[str | None, IDENTITY] = None
    target_layer_rules: Annotated[tuple[str, ...], IDENTITY] = ()
    exclusions: Annotated[tuple[str, ...], IDENTITY] = ()
    producer: Annotated[str | None, IDENTITY] = None
    version: Annotated[str | None, IDENTITY] = None
