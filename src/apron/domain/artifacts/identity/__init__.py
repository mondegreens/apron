"""Layer 1 — ArtifactIdentity (ADR-002 §8, ADR-006).

Identity is derived from resolved file/component contents and structure,
not from a registry namespace.  Two observations with identical content
digests share the same ArtifactIdentity.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict

from apron.domain.artifacts import ArtifactSourceObservation
from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.fingerprints import (
    DISPLAY,
    IDENTITY,
)


class ArtifactIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    content_digest: Annotated[str, IDENTITY]
    component_structure_digest: Annotated[str | None, IDENTITY] = None

    @classmethod
    def from_observation(
        cls,
        obs: ArtifactSourceObservation,
        component_structure_digest: str | None = None,
    ) -> Self:
        sorted_digests = sorted([fd.path, fd.sha256] for fd in obs.file_digests)
        content_digest = digest_hex(canonicalize(sorted_digests))
        return cls(
            content_digest=content_digest,
            component_structure_digest=component_structure_digest,
        )
