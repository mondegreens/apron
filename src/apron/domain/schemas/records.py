"""Layer 5 — Evidentiary records, diagnosis rules, and release manifests.

Every evidentiary record (TaskAttemptRecord, VerificationReport,
RemediationRecord) carries claim_scope, production_mode, reason (INV-20),
lifecycle, and corrects (§2.0 correction convention).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict

from apron.domain.fingerprints import DISPLAY, IDENTITY, FingerprintHex
from apron.domain.schemas.migrations import register
from apron.domain.schemas.primitives import ClaimScope

# ---------------------------------------------------------------------------
# TaskAttemptRecord (ADR-011 §7, ADR-010 §15)
# ---------------------------------------------------------------------------


class TaskAttemptRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    # What was verified: the solution and the exact plan that was booted.
    solution_fingerprint: Annotated[FingerprintHex | None, IDENTITY] = None
    deployment_plan_digest: Annotated[FingerprintHex | None, IDENTITY] = None
    boot_outcome: Annotated[Literal["healthy", "failed"] | None, IDENTITY] = None
    failures: Annotated[tuple[str, ...], DISPLAY] = ()
    log_tail: Annotated[str | None, DISPLAY] = None
    # Prediction delta (M5 sign convention: predicted minus measured, bytes).
    predicted_minus_measured: Annotated[dict[str, int] | None, DISPLAY] = None
    prediction_notes: Annotated[tuple[str, ...], DISPLAY] = ()
    # Cost method (§7): rate, local phase timing, estimate and provider-reported cost.
    hourly_rate: Annotated[float | None, DISPLAY] = None
    phase_seconds: Annotated[dict[str, float] | None, DISPLAY] = None
    estimated_cost: Annotated[float | None, DISPLAY] = None
    provider_reported_cost: Annotated[float | None, DISPLAY] = None
    # Serving measurement (claim_scope="serving_performance"); measurement only.
    serving_input_sequence_length: Annotated[int | None, DISPLAY] = None
    serving_output_sequence_length: Annotated[int | None, DISPLAY] = None
    serving_concurrency: Annotated[int | None, DISPLAY] = None
    serving_num_prompts: Annotated[int | None, DISPLAY] = None
    serving_completed: Annotated[int | None, DISPLAY] = None
    serving_failed: Annotated[int | None, DISPLAY] = None
    serving_latency_ms: Annotated[dict[str, float] | None, DISPLAY] = None
    serving_request_throughput: Annotated[float | None, DISPLAY] = None
    serving_output_token_throughput: Annotated[float | None, DISPLAY] = None
    serving_slo_verdict: Annotated[Literal["pass", "fail", "none_declared"] | None, DISPLAY] = None
    serving_slo_reasons: Annotated[tuple[str, ...], DISPLAY] = ()
    claim_scope: Annotated[ClaimScope, IDENTITY]
    production_mode: Annotated[bool, IDENTITY]
    reason: Annotated[str, IDENTITY]
    lifecycle: Annotated[Literal["observed", "retracted"], IDENTITY]
    corrects: Annotated[str | None, IDENTITY] = None


# ---------------------------------------------------------------------------
# RemediationRecord (ADR-006 §Remediation-proof, §Failure-fingerprint)
# ---------------------------------------------------------------------------


class RemediationRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    # Classifier provenance (F6): a classifier change is a new diagnosis input.
    classifier_model_id: Annotated[str | None, IDENTITY] = None
    classifier_input_digest: Annotated[FingerprintHex | None, IDENTITY] = None
    diagnosed_failure_class: Annotated[str | None, IDENTITY] = None
    classifier_evidence_span: Annotated[str | None, DISPLAY] = None
    classifier_extraction: Annotated[dict[str, int | float | str], DISPLAY] = {}
    correction_strategy: Annotated[str | None, DISPLAY] = None
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
#
# The typed form of a rule file under ``rules/<engine>-v<major.minor>/``
# (Phase 1b §10.2 step 8: the JSON shape and this schema were reconciled).
# Rule files are loaded through the migration path and validated strictly.


class SourceSite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    file: Annotated[str, IDENTITY]
    line: Annotated[int, IDENTITY]


class ExtractionField(BaseModel):
    """One value diagnosis extracts from an error (INV-6).

    Curated class rules declare typed numerics and enums, each citing the
    engine source line that prints it.  Scanner-generated hypothesis rules
    may still declare ``string`` fields without a source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Annotated[Literal["int", "float", "enum", "string"], IDENTITY]
    enum: Annotated[tuple[str, ...], IDENTITY] = ()
    source: Annotated[str | None, IDENTITY] = None


class DiagnosisRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 4
    engine: Annotated[str, IDENTITY]
    engine_version: Annotated[str, IDENTITY]
    error_family: Annotated[str, IDENTITY]
    rule_version: Annotated[int, IDENTITY] = 1
    supersedes: Annotated[FingerprintHex | None, IDENTITY] = None
    correction_strategy: Annotated[str | None, IDENTITY] = None
    correction_spec: Annotated[dict[str, str], IDENTITY] = {}
    extraction_schema: Annotated[dict[str, ExtractionField], IDENTITY] = {}
    source_sites: Annotated[tuple[SourceSite, ...], IDENTITY] = ()
    float16_blocklist: Annotated[tuple[str, ...], IDENTITY] = ()
    float16_blocklist_source: Annotated[str | None, DISPLAY] = None
    kernel_architectures: Annotated[dict[str, tuple[str, ...]], IDENTITY] = {}
    kernel_architectures_source: Annotated[str | None, DISPLAY] = None
    exception_class: Annotated[str | None, IDENTITY] = None
    engine_callsite_module: Annotated[str | None, IDENTITY] = None
    status: Annotated[
        Literal["hypothesis", "mechanism_verified"],
        IDENTITY,
    ] = "hypothesis"
    promoting_verification_fingerprint: Annotated[FingerprintHex | None, DISPLAY] = None
    examples: Annotated[tuple[str, ...], DISPLAY] = ()
    count: Annotated[int | None, DISPLAY] = None
    curation: Annotated[str | None, DISPLAY] = None
    replaces: Annotated[tuple[str, ...], DISPLAY] = ()


@register("DiagnosisRule", 3)
def _migrate_rule_v3_to_v4(data: dict[str, object]) -> dict[str, object]:
    """v3 rule files: plain type strings become ExtractionField objects;
    rules gain ``rule_version`` 1."""
    raw = data.get("extraction_schema") or {}
    schema = raw if isinstance(raw, dict) else {}
    return {
        **data,
        "schema_version": 4,
        "rule_version": data.get("rule_version", 1),
        "extraction_schema": {
            name: ({"type": spec} if isinstance(spec, str) else spec)
            for name, spec in schema.items()
        },
    }


# ---------------------------------------------------------------------------
# EvidenceReleaseManifest (ADR-002 §6)
# ---------------------------------------------------------------------------


class EvidenceReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Annotated[int, DISPLAY] = 1
    record_digests: Annotated[tuple[str, ...], IDENTITY]
    signatures: Annotated[tuple[str, ...], IDENTITY] = ()
    attestations: Annotated[tuple[str, ...], IDENTITY] = ()
    license: Annotated[str, IDENTITY] = "CDLA-Permissive-2.0"
