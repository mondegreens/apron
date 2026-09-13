"""Layer 4 — Solution, plan, and evaluation schemas."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex
from apron.domain.schemas.primitives import ClaimScope

# ---------------------------------------------------------------------------
# PlanningClaim (ADR-002 §10)
# ---------------------------------------------------------------------------


class PlanningClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    producer: Annotated[str, IDENTITY]
    version: Annotated[str, IDENTITY]
    input_fingerprint: Annotated[str, IDENTITY]
    proposed_configuration: Annotated[dict[str, Any], IDENTITY]
    claim_scope: Annotated[ClaimScope, IDENTITY]
    producer_epistemic_tier: Annotated[str, IDENTITY]
    source_record_references: Annotated[tuple[str, ...], IDENTITY] = ()
    calibration_domain: Annotated[str | None, IDENTITY] = None
    uncertainty: Annotated[dict[str, float] | None, DISPLAY] = None
    unsupported_fields: Annotated[tuple[str, ...], DISPLAY] = ()
    opaque_fields: Annotated[tuple[str, ...], DISPLAY] = ()


# ---------------------------------------------------------------------------
# DeploymentPlan (phase-plan §Phase 0)
# ---------------------------------------------------------------------------


class DeploymentPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    tensor_parallel: Annotated[int, IDENTITY] = 1
    pipeline_parallel: Annotated[int, IDENTITY] = 1
    expert_parallel: Annotated[int, IDENTITY] = 1
    data_parallel: Annotated[int, IDENTITY] = 1
    disaggregated_prefill: Annotated[bool, IDENTITY] = False
    disaggregated_decode: Annotated[bool, IDENTITY] = False
    engine_configuration: Annotated[dict[str, str], IDENTITY] = {}
    batch_size: Annotated[int | None, IDENTITY] = None
    dtype: Annotated[str | None, IDENTITY] = None
    resource_allocation: Annotated[dict[str, str], IDENTITY] = {}
    serve_command: Annotated[str | None, DISPLAY] = None
    docker_compose: Annotated[str | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# EvaluationProtocol (ADR-011 §5)
# ---------------------------------------------------------------------------


class EvaluationProtocol(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    decision_request_digest: Annotated[str, IDENTITY]
    task_suite_fingerprint: Annotated[FingerprintHex, IDENTITY]
    application_fingerprint: Annotated[FingerprintHex, IDENTITY]
    solution_fingerprint: Annotated[FingerprintHex, IDENTITY]
    harness: Annotated[str, IDENTITY]
    harness_version: Annotated[str, IDENTITY]
    dataset_snapshot: Annotated[str | None, IDENTITY] = None
    scorer: Annotated[str, IDENTITY]
    rubric: Annotated[str | None, IDENTITY] = None
    deterministic_checks: Annotated[tuple[str, ...], IDENTITY] = ()
    judge_model: Annotated[str | None, IDENTITY] = None
    judge_prompt: Annotated[str | None, IDENTITY] = None
    sampling_temperature: Annotated[float | None, IDENTITY] = None
    sampling_top_p: Annotated[float | None, IDENTITY] = None
    seeds: Annotated[tuple[int, ...], IDENTITY] = ()
    repetitions: Annotated[int, IDENTITY] = 1
    concurrency: Annotated[int | None, IDENTITY] = None
    stopping_rules: Annotated[tuple[str, ...], IDENTITY] = ()
    aggregation_method: Annotated[str | None, IDENTITY] = None
    uncertainty_method: Annotated[str | None, IDENTITY] = None
