"""Layer 3 — Task, application, and serving schemas.

TaskSuiteSpec and ServingWorkloadSpec are independent (INV-24): serving
throughput cannot establish task success.  WorkloadSpec is a compatibility
envelope that binds them by fingerprint reference.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex  # noqa: TC001


# ---------------------------------------------------------------------------
# TaskSuiteSpec (ADR-011 §2)
# ---------------------------------------------------------------------------


class TaskSuiteSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    name: Annotated[str, IDENTITY]
    version: Annotated[str, IDENTITY]
    required_capabilities: Annotated[tuple[CapabilitySignature, ...], IDENTITY]
    cases: Annotated[tuple[dict[str, str], ...], IDENTITY] = ()
    sampling_frame: Annotated[str | None, IDENTITY] = None
    tools: Annotated[tuple[str, ...], IDENTITY] = ()
    environment: Annotated[str | None, IDENTITY] = None
    task_categories: Annotated[tuple[str, ...], IDENTITY] = ()
    success_criteria: Annotated[tuple[str, ...], IDENTITY] = ()
    rubrics: Annotated[tuple[str, ...], IDENTITY] = ()
    segment_weights: Annotated[dict[str, float] | None, IDENTITY] = None
    privacy_classification: Annotated[str, IDENTITY] = "public"
    representativeness_claims: Annotated[tuple[str, ...], DISPLAY] = ()


# ---------------------------------------------------------------------------
# ApplicationSpec (ADR-011 §3)
# ---------------------------------------------------------------------------


class ApplicationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    name: Annotated[str, IDENTITY]
    version: Annotated[str, IDENTITY]
    prompt_template_revision: Annotated[str | None, IDENTITY] = None
    agent_scaffold_version: Annotated[str | None, IDENTITY] = None
    tool_definitions: Annotated[tuple[str, ...], IDENTITY] = ()
    tool_versions: Annotated[dict[str, str], IDENTITY] = {}
    retrieval_components: Annotated[tuple[str, ...], IDENTITY] = ()
    reasoning_settings: Annotated[dict[str, str], IDENTITY] = {}
    sampling_settings: Annotated[dict[str, str], IDENTITY] = {}
    turn_limit: Annotated[int | None, IDENTITY] = None
    tool_call_limit: Annotated[int | None, IDENTITY] = None
    retry_limit: Annotated[int | None, IDENTITY] = None
    logical_roles: Annotated[tuple[str, ...], IDENTITY] = ()
    invocation_conditions: Annotated[dict[str, str], IDENTITY] = {}


# ---------------------------------------------------------------------------
# ServingWorkloadSpec (ADR-011 §2)
# ---------------------------------------------------------------------------


class ServingWorkloadSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    request_distribution: Annotated[str | None, IDENTITY] = None
    concurrency: Annotated[int | None, IDENTITY] = None
    burstiness: Annotated[float | None, IDENTITY] = None
    input_shape_distribution: Annotated[str | None, IDENTITY] = None
    output_shape_distribution: Annotated[str | None, IDENTITY] = None
    prefix_cache_behavior: Annotated[str | None, IDENTITY] = None
    lora_usage: Annotated[str | None, IDENTITY] = None
    latency_p99_ms: Annotated[float | None, IDENTITY] = None
    throughput_target_rps: Annotated[float | None, IDENTITY] = None
    availability_target: Annotated[float | None, IDENTITY] = None
    duration_hours: Annotated[float | None, IDENTITY] = None
    capacity_horizon: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# WorkloadSpec (ADR-011 §2) — compatibility envelope
# ---------------------------------------------------------------------------


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    task_suite_fingerprint: Annotated[FingerprintHex, IDENTITY]
    serving_workload_fingerprint: Annotated[FingerprintHex, IDENTITY]
