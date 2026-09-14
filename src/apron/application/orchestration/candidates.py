"""Candidate graph — enumerate resolved and legacy candidates.

A CandidateGraph holds ResolvedCandidates from both the resolution
chain and from legacy CatalogueImportSource entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from apron.domain.schemas.imports import ImportedBootObservation
from apron.domain.schemas.models import ArtifactSpec, ExecutionSpec


@dataclass(frozen=True)
class ResolvedCandidate:
    artifact_spec: ArtifactSpec
    execution_spec: ExecutionSpec | None = None
    source: Literal["resolved", "legacy"] = "resolved"
    unsupported_reason: str | None = None

    @property
    def deployment_feasible(self) -> bool:
        if self.source == "legacy":
            return False
        return self.execution_spec is not None


@dataclass
class CandidateGraph:
    candidates: list[ResolvedCandidate] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        resolved: ArtifactSpec | None = None,
        execution_spec: ExecutionSpec | None = None,
        legacy_entries: list[dict[str, Any]] | None = None,
    ) -> CandidateGraph:
        candidates: list[ResolvedCandidate] = []

        if resolved is not None:
            candidates.append(
                ResolvedCandidate(
                    artifact_spec=resolved,
                    execution_spec=execution_spec,
                    source="resolved",
                )
            )

        for entry in legacy_entries or []:
            if isinstance(entry, ImportedBootObservation):
                obs = entry
            else:
                obs = ImportedBootObservation(**entry)
            legacy_spec = _legacy_to_artifact_spec(obs)
            candidates.append(
                ResolvedCandidate(
                    artifact_spec=legacy_spec,
                    execution_spec=None,
                    source="legacy",
                )
            )

        return cls(candidates=candidates)


def _legacy_to_artifact_spec(obs: ImportedBootObservation) -> ArtifactSpec:
    """Convert a legacy ImportedBootObservation to an ArtifactSpec."""
    from apron.domain.artifacts.identity import ArtifactIdentity
    from apron.domain.canonical import digest_hex
    from apron.domain.schemas.primitives import ArtifactLocator

    content_digest = digest_hex(f"legacy:{obs.model_id}".encode())
    identity = ArtifactIdentity(content_digest=content_digest)
    locator = ArtifactLocator(source_kind="legacy", uri=obs.model_id)

    return ArtifactSpec(
        identity=identity,
        locators=(locator,),
    )
