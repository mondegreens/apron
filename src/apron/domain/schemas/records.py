"""Layer 5 — Evidentiary records, diagnosis rules, and release manifests.

Every evidentiary record (TaskAttemptRecord, VerificationReport,
RemediationRecord) carries claim_scope, production_mode, reason (INV-20),
lifecycle, and corrects (§2.0 correction convention).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex  # noqa: TC001
from apron.domain.schemas.primitives import ClaimScope


# ---------------------------------------------------------------------------
# TaskAttemptRecord (ADR-011 §7, ADR-010 §15)
# ---------------------------------------------------------------------------


class TaskAttemptRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    decision_fingerprint: Annotated[FingerprintHex, IDENTITY]
    task_suite_fingerprint: Annotated[FingerprintHex, IDENTITY]
    application_fingerprint: Annotated[FingerprintHex, IDENTITY]
    evaluation_protocol_fingerprint: Annotated[FingerprintHex, IDENTITY]
    solution_fingerprint: Annotated[FingerprintHex, IDENTITY]
    case_id: Annotated[str, IDENTITY]
    attempt_id: Annotated[str, IDENTITY]
    output: Annotated[str | None, DISPLAY] = None
    permitted_artifacts: Annotated[tuple[str, ...], DISPLAY] = ()
    criterion_scores: Annotated[dict[str, float], DISPLAY] = {}
    accepted: Annotated[bool | None, DISPLAY] = None
    trace_references: Annotated[tuple[str, ...], DISPLAY] = ()
    turns: Annotated[int | None, DISPLAY] = None
    retries: Annotated[int | None, DISPLAY] = None
    tool_calls: Annotated[int | None, DISPLAY] = None
    input_tokens: Annotated[int | None, DISPLAY] = None
    cached_input_tokens: Annotated[int | None, DISPLAY] = None
    output_tokens: Annotated[int | None, DISPLAY] = None
    time_seconds: Annotated[float | None, DISPLAY] = None
    endpoint_cost: Annotated[float | None, DISPLAY] = None
    judge_cost: Annotated[float | None, DISPLAY] = None
    infrastructure_cost: Annotated[float | None, DISPLAY] = None
    failures: Annotated[tuple[str, ...], DISPLAY] = ()
    raw_observation_provenance: Annotated[str | None, DISPLAY] = None
    market_equivalent_price: Annotated[float | None, DISPLAY] = None
    gross_attributable_cost: Annotated[float | None, DISPLAY] = None
    subsidy_applied: Annotated[float | None, DISPLAY] = None
    project_out_of_pocket_cost: Annotated[float | None, DISPLAY] = None
    claim_scope: Annotated[ClaimScope, IDENTITY]
    production_mode: Annotated[bool, IDENTITY]
    reason: Annotated[str, IDENTITY]
    lifecycle: Annotated[Literal["observed", "retracted"], IDENTITY]
    corrects: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# VerificationReport (ADR-006, phase-plan, ADR-010 §15)
# ---------------------------------------------------------------------------


class VerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    target_kind: Annotated[str, IDENTITY]
    operator: Annotated[str, IDENTITY]
    provider: Annotated[str | None, IDENTITY] = None
    detected_hardware_fingerprint: Annotated[str | None, IDENTITY] = None
    execution_fingerprint: Annotated[str, IDENTITY]
    initial_total_memory: Annotated[int | None, DISPLAY] = None
    initial_free_memory: Annotated[int | None, DISPLAY] = None
    requested_memory: Annotated[int | None, DISPLAY] = None
    model_weight_memory: Annotated[int | None, DISPLAY] = None
    persistent_consumption: Annotated[int | None, DISPLAY] = None
    transient_peak_headroom: Annotated[int | None, DISPLAY] = None
    non_pytorch_increase: Annotated[int | None, DISPLAY] = None
    cuda_graph_estimate: Annotated[int | None, DISPLAY] = None
    cuda_graph_applied: Annotated[int | None, DISPLAY] = None
    cuda_graph_actual: Annotated[int | None, DISPLAY] = None
    available_kv_cache_memory: Annotated[int | None, DISPLAY] = None
    safety_buffer: Annotated[int | None, DISPLAY] = None
    profiling_shape: Annotated[dict[str, int] | None, DISPLAY] = None
    market_equivalent_price: Annotated[float | None, DISPLAY] = None
    gross_attributable_cost: Annotated[float | None, DISPLAY] = None
    subsidy_applied: Annotated[float | None, DISPLAY] = None
    project_out_of_pocket_cost: Annotated[float | None, DISPLAY] = None
    claim_scope: Annotated[ClaimScope, IDENTITY]
    production_mode: Annotated[bool, IDENTITY]
    reason: Annotated[str, IDENTITY]
    lifecycle: Annotated[Literal["observed", "retracted"], IDENTITY]
    corrects: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# RemediationRecord (ADR-006 §Remediation-proof, §Failure-fingerprint)
# ---------------------------------------------------------------------------


class RemediationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    mechanism_outcome: Annotated[
        Literal["verified", "failed", "not_evaluated"],
        IDENTITY,
    ]
    request_outcome: Annotated[
        Literal["satisfied", "violated", "not_evaluated"],
        IDENTITY,
    ]
    accepted_request_digest: Annotated[str, IDENTITY]
    task_fingerprint: Annotated[FingerprintHex, IDENTITY]
    application_fingerprint: Annotated[FingerprintHex, IDENTITY]
    evaluation_fingerprint: Annotated[FingerprintHex, IDENTITY]
    corrected_plan_digest: Annotated[str, IDENTITY]
    corrected_solution_digest: Annotated[str | None, IDENTITY] = None
    violated_constraints: Annotated[tuple[str, ...], DISPLAY] = ()
    proving_record_fingerprints: Annotated[tuple[FingerprintHex, ...], DISPLAY] = ()
    claim_scope: Annotated[Literal["remediation"], IDENTITY] = "remediation"
    production_mode: Annotated[bool, IDENTITY] = False
    reason: Annotated[str, IDENTITY] = ""
    lifecycle: Annotated[Literal["observed", "retracted"], IDENTITY] = "observed"
    corrects: Annotated[str | None, IDENTITY] = None


def derive_remediation_result(
    mechanism_outcome: str,
    request_outcome: str,
) -> str:
    if mechanism_outcome == "verified" and request_outcome == "satisfied":
        return "Fixed"
    if mechanism_outcome == "verified" and request_outcome == "violated":
        return "Alternative with trade-offs"
    return "Unverified suggestion"


# ---------------------------------------------------------------------------
# DiagnosisRule (ADR-006 §Failure-fingerprint, §Remediation-proof)
# ---------------------------------------------------------------------------


class DiagnosisRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    exception_class: Annotated[str, IDENTITY]
    engine_callsite_module: Annotated[str, IDENTITY]
    error_family: Annotated[str, IDENTITY]
    correction_spec: Annotated[dict[str, str], IDENTITY] = {}
    status: Annotated[
        Literal["hypothesis", "mechanism_verified"],
        IDENTITY,
    ] = "hypothesis"
    promoting_verification_fingerprint: Annotated[FingerprintHex | None, DISPLAY] = None


# ---------------------------------------------------------------------------
# EvidenceReleaseManifest (ADR-002 §6)
# ---------------------------------------------------------------------------


class EvidenceReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[int, DISPLAY] = 1
    record_digests: Annotated[tuple[str, ...], IDENTITY]
    signatures: Annotated[tuple[str, ...], IDENTITY] = ()
    attestations: Annotated[tuple[str, ...], IDENTITY] = ()
    license: Annotated[str, IDENTITY] = "CDLA-Permissive-2.0"
