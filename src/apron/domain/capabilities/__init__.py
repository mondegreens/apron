"""Layer 0 — CapabilitySignature (ADR-007 §2).

A capability signature describes what an endpoint can do: which input
modalities are required together, what output representation is produced,
and whether the operation streams.

``required_inputs`` is a tuple of modality names.  ``("text", "image")``
means both are required together (T+I); a model that also accepts text
alone would have a SECOND CapabilitySignature with ``("text",)``.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import (
    DISPLAY,
    IDENTITY,
)


class CapabilitySignature(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    operation: Annotated[str, IDENTITY]
    required_inputs: Annotated[tuple[str, ...], IDENTITY]
    output_representation: Annotated[str, IDENTITY]
    streaming: Annotated[bool, IDENTITY] = False
