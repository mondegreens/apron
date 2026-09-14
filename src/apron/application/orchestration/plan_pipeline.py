"""Plan pipeline — orchestrate resolution → calculator → plan → render.

Extracted from CLI to satisfy INV-11: no recommendation, calculation,
ranking or promotion logic in interfaces.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.plan_builder import build_plan
from apron.application.orchestration.resolution import ResolutionChain
from apron.domain.mechanisms.model_spec_builder import build_model_spec
from apron.domain.schemas.solutions import DeploymentPlan, RenderContext

if TYPE_CHECKING:
    from apron.domain.ports import Clock, IdGenerator
    from apron.domain.protocols import ArtifactSourceResolver
    from apron.domain.schemas.primitives import HardwareSpec


class PlanPipelineResult:
    __slots__ = ("plan", "context", "claim", "error")

    def __init__(
        self,
        *,
        plan: DeploymentPlan | None = None,
        context: RenderContext | None = None,
        claim: Any = None,
        error: str | None = None,
    ) -> None:
        self.plan = plan
        self.context = context
        self.claim = claim
        self.error = error

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None


def run_plan_pipeline(
    resolver: ArtifactSourceResolver,
    model_id: str,
    hardware: HardwareSpec,
    *,
    clock: Clock,
    id_gen: IdGenerator,
) -> PlanPipelineResult:
    """Run the full plan pipeline: resolve → calculate → build plan."""
    chain = ResolutionChain(resolver)
    result = chain.resolve(model_id)

    if not result.ok:
        return PlanPipelineResult(error=str(result.error))

    assert result.observation is not None

    config_content = _download_config(resolver, model_id, result.observation.resolved_revision)
    if config_content is None:
        return PlanPipelineResult(error="config.json not found")

    config = json.loads(config_content)

    total_weight_bytes = _resolve_weight_bytes(resolver, model_id, result.observation)

    model_spec = build_model_spec(
        config,
        repository=model_id,
        revision=result.observation.resolved_revision,
        license_id=result.observation.license_observed,
    )

    calc_metadata = dict(config)
    calc_metadata["total_weight_bytes"] = total_weight_bytes
    calc_metadata["components"] = list(model_spec.components)

    from apron.adapters.planning.calculator_source import CalculatorPlanningSource

    planning_source = CalculatorPlanningSource(clock=clock)
    claim = planning_source.predict(
        calc_metadata, hardware, {"isl": 512, "osl": 128, "max_batch_size": 4}
    )

    if claim.proposed_configuration.get("status") == "unknown":
        return PlanPipelineResult(
            error="unknown model mechanism — calculator cannot predict memory"
        )

    deployment_plan = build_plan(
        claim,
        model_spec,
        hardware,
        result.execution_spec,
        None,
        clock=clock,
        id_gen=id_gen,
    )

    assert result.locator is not None
    ctx = RenderContext(plan=deployment_plan, locator=result.locator, hardware=hardware)

    return PlanPipelineResult(plan=deployment_plan, context=ctx, claim=claim)


def _download_config(resolver: Any, model_id: str, revision: str) -> bytes | None:
    if hasattr(resolver, "_download_file"):
        return resolver._download_file(model_id, "config.json", revision)
    return None


def _resolve_weight_bytes(resolver: Any, model_id: str, observation: Any) -> int:
    rev = observation.resolved_revision
    total_weight_bytes = 0

    if hasattr(resolver, "_download_file"):
        index_content = resolver._download_file(model_id, "model.safetensors.index.json", rev)
        if index_content is not None:
            index_data = json.loads(index_content)
            total_weight_bytes = index_data.get("metadata", {}).get("total_size", 0)

    if total_weight_bytes == 0:
        safetensors_params = observation.publisher_metadata or {}
        for key, val in safetensors_params.items():
            if key.startswith("parameters_"):
                dtype_suffix = key.split("_", 1)[1]
                bytes_per_param = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}.get(dtype_suffix, 2)
                total_weight_bytes = int(val) * bytes_per_param
                break

    return total_weight_bytes
