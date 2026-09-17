"""Resolution chain — resolves a model reference through HF Hub to InferenceSolution.

Each step returns a typed StepResult — success or failure with reason.
Resolution does not crash; a failed step produces a typed error and halts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apron.domain.artifacts.identity import ArtifactIdentity
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.models import ArtifactSpec, ExecutionSpec
from apron.domain.schemas.primitives import ArtifactLocator
from apron.domain.solutions import DirectEndpoint, InferenceSolution

if TYPE_CHECKING:
    from apron.domain.artifacts import ArtifactSourceObservation
    from apron.domain.protocols import ArtifactSourceResolver, EvidenceSource

__all__ = [
    "ResolutionChain",
    "ResolutionResult",
    "StepError",
    "StepResult",
]


@dataclass(frozen=True)
class StepError:
    step: str
    reason: str
    model_id: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class StepResult[T]:
    value: T | None = None
    error: StepError | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.error is None


@dataclass(frozen=True)
class ResolutionResult:
    locator: ArtifactLocator | None = None
    observation: ArtifactSourceObservation | None = None
    identity: ArtifactIdentity | None = None
    artifact_spec: ArtifactSpec | None = None
    execution_spec: ExecutionSpec | None = None
    solution: InferenceSolution | None = None
    error: StepError | None = None

    @property
    def ok(self) -> bool:
        return self.solution is not None and self.error is None


class ResolutionChain:
    """Construct the full identity chain from a model reference.

    Accepts adapters through Protocol injection (import-linter:
    application cannot import adapters).
    """

    def __init__(
        self,
        resolver: ArtifactSourceResolver,
        evidence_sources: list[EvidenceSource] | None = None,
    ) -> None:
        self._resolver = resolver
        self._evidence_sources = evidence_sources or []

    def resolve(
        self,
        model_id: str,
        source_kind: str = "huggingface",
        revision: str | None = None,
        engine_constraints: dict[str, Any] | None = None,
    ) -> ResolutionResult:
        # Step 1: ArtifactLocator
        locator = ArtifactLocator(
            source_kind=source_kind,
            uri=model_id,
            requested_revision=revision,
        )

        # Step 2: ArtifactSourceObservation
        if not self._resolver.supports(source_kind):
            return ResolutionResult(
                locator=locator,
                error=StepError(
                    step="resolve_artifact",
                    reason=f"unsupported source kind: {source_kind}",
                    model_id=model_id,
                    source=source_kind,
                ),
            )

        try:
            observation = self._resolver.resolve(locator)
        except Exception as exc:
            return ResolutionResult(
                locator=locator,
                error=StepError(
                    step="resolve_artifact",
                    reason=str(exc),
                    model_id=model_id,
                    source=source_kind,
                ),
            )

        # Step 3: ArtifactIdentity
        identity = ArtifactIdentity.from_observation(observation)

        # Step 4: ArtifactSpec — use file content SHA-256 directly (no double-hash)
        config_digest = None
        weight_manifest_digest = None
        for fd in observation.file_digests:
            if fd.path == "config.json":
                config_digest = fd.sha256
            elif fd.path == "model.safetensors.index.json":
                weight_manifest_digest = fd.sha256

        artifact_spec = ArtifactSpec(
            identity=identity,
            locators=(locator,),
            observations=(observation,),
            config_digest=config_digest,
            weight_manifest_digest=weight_manifest_digest,
        )

        # Step 5: ExecutionSpec (from engine constraints)
        execution_spec = None
        if engine_constraints is not None:
            execution_spec = ExecutionSpec(
                engine_image_digest=engine_constraints.get("engine_image_digest", "unknown"),
                resolved_checkpoint_method=engine_constraints.get(
                    "resolved_checkpoint_method", "safetensors"
                ),
            )

        # Step 6: InferenceSolution with one DirectEndpoint
        model_spec_fp = fingerprint_hex(artifact_spec)
        artifact_spec_fp = fingerprint_hex(artifact_spec)
        execution_spec_fp = fingerprint_hex(execution_spec) if execution_spec else None

        endpoint = DirectEndpoint(
            binding="direct_endpoint",
            model_spec_fingerprint=model_spec_fp,
            artifact_spec_fingerprint=artifact_spec_fp,
            execution_spec_fingerprint=execution_spec_fp,
            deployment_plan_fingerprint=None,
        )

        solution = InferenceSolution(endpoints=(endpoint,))

        return ResolutionResult(
            locator=locator,
            observation=observation,
            identity=identity,
            artifact_spec=artifact_spec,
            execution_spec=execution_spec,
            solution=solution,
        )
