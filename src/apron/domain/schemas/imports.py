"""Imported boot observations from external deployment catalogues.

External boot data enters Apron as seed candidates — models attested to
run on certain hardware with observed VRAM and a working engine
configuration, but without Apron's own execution fingerprint, profiled
memory breakdown, or verified task evidence.

``import_status`` separates these from Apron-produced claims.  It is not
an ``EpistemicStatus`` variant and not a position in the evidence total
order.  It marks the record as externally attested data that seeds the
candidate pool but cannot satisfy any verification gate until Apron
replays the boot.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import DISPLAY, IDENTITY

ImportStatus = Literal["owner_attested_boot"]


class ImportedBootObservation(BaseModel):
    """A boot observation from an external deployment catalogue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    import_status: Annotated[ImportStatus, IDENTITY] = "owner_attested_boot"
    model_id: Annotated[str, IDENTITY]
    outcome: Annotated[str, IDENTITY] = "booted_healthy"
    provider: Annotated[str, IDENTITY] = "runpod"
    empirical_vram_gb: Annotated[float | None, IDENTITY] = None
    observed_configuration: Annotated[dict[str, Any], IDENTITY] = {}
    provenance: Annotated[str, IDENTITY] = ""
    artifact_revision: Annotated[None, DISPLAY] = None
    engine_image_digest: Annotated[None, DISPLAY] = None
    execution_fingerprint: Annotated[None, DISPLAY] = None
    hardware_fingerprint: Annotated[None, DISPLAY] = None
    workload: Annotated[None, DISPLAY] = None
    date: Annotated[None, DISPLAY] = None


def _is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


class CatalogueImportSource:
    """EvidenceSource adapter that reads an external model registry.

    Filters out excluded entries (Gemma).  Preserves empirical VRAM and
    the working engine configuration.  Does not carry forward model
    metadata (parameter counts, architecture labels, hidden dimensions)
    — those are re-resolved from immutable artifacts.  Does not carry
    forward generic estimation formulas.
    """

    def __init__(
        self,
        registry: dict[str, dict[str, Any]],
        *,
        source_name: str,
        provenance: str,
    ) -> None:
        self._registry = registry
        self._source_name = source_name
        self._provenance = provenance

    @property
    def source_name(self) -> str:
        return self._source_name

    def collect(self) -> list[dict[str, Any]]:
        entries = []
        for model_id, row in self._registry.items():
            if _is_gemma(model_id):
                continue
            entries.append(
                ImportedBootObservation(
                    model_id=model_id,
                    empirical_vram_gb=row.get("vram_fp16_gb"),
                    observed_configuration=row.get("vllm_params", {}),
                    provenance=self._provenance,
                ).model_dump(mode="json")
            )
        return entries
