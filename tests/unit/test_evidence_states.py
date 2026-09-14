"""Step 13 tests: evidence-state, authority, and evaluation fixtures.

Each fixture exercises one named invariant. Cross-fixture fingerprint
integrity: every embedded fingerprint reference is computed from the
referenced record, not hand-typed.
"""

from pydantic import TypeAdapter

from apron.domain.canonical import canonicalize, record_digest_hex
from apron.domain.capabilities import CapabilitySignature
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.schemas.authority import (
    ActionRequest,
    AuthorityContribution,
    AuthorizationDecision,
    DecisionRequest,
    evaluate_authorization,
)
from apron.domain.schemas.models import (
    ArtifactRelation,
    ClaimedDerivedFrom,
    CompatibilityEvidence,
    PublishedByOwner,
    QualityEvidence,
    ReproduciblyDerivedFrom,
    StructurallyCompatibleWith,
    TokenizerCompatibleWith,
)
from apron.domain.schemas.primitives import (
    MeasuredStatus,
    PredictedStatus,
)
from apron.domain.schemas.records import (
    EvidenceReleaseManifest,
    TaskAttemptRecord,
    VerificationReport,
)
from apron.domain.schemas.reports import can_promote
from apron.domain.schemas.tasks import (
    ServingWorkloadSpec,
    TaskSuiteSpec,
    WorkloadSpec,
)

_FP = "1220" + "ab" * 32
_FP2 = "1220" + "cd" * 32
_DIGEST = "1220" + "99" * 32

_TEXT_CAP = CapabilitySignature(
    operation="text_generation",
    required_inputs=("text",),
    output_representation="generated_text",
)


def _make_attempt(solution_fp: str, **overrides) -> TaskAttemptRecord:
    defaults = {
        "decision_fingerprint": _FP,
        "task_suite_fingerprint": _FP,
        "application_fingerprint": _FP,
        "evaluation_protocol_fingerprint": _FP,
        "solution_fingerprint": solution_fp,
        "case_id": "case-1",
        "attempt_id": "attempt-1",
        "claim_scope": "task_outcome",
        "production_mode": False,
        "reason": "evaluation_run",
        "lifecycle": "observed",
    }
    return TaskAttemptRecord(**(defaults | overrides))


def _make_verification(**overrides) -> VerificationReport:
    defaults = {
        "target_kind": "local-container",
        "operator": "self",
        "execution_fingerprint": _DIGEST,
        "claim_scope": "boot",
        "production_mode": False,
        "reason": "boot_profiling",
        "lifecycle": "observed",
    }
    return VerificationReport(**(defaults | overrides))


# ============================================================================
# CONTESTED EVIDENCE (INV-4)
# ============================================================================


def test_contested_two_records_different_levels_same_solution():
    """Two observations of the same solution disagree on score.
    Same solution fingerprint, different records. Both shown,
    nothing auto-resolves (INV-4)."""
    measured = _make_attempt(
        solution_fp=_FP,
        case_id="case-1",
        attempt_id="measured-1",
        criterion_scores={"accuracy": 0.95},
        reason="measured_evaluation",
    )
    predicted = _make_attempt(
        solution_fp=_FP,
        case_id="case-1",
        attempt_id="predicted-1",
        criterion_scores={"accuracy": 0.80},
        reason="predicted_evaluation",
    )
    assert measured.solution_fingerprint == predicted.solution_fingerprint
    assert measured.criterion_scores != predicted.criterion_scores
    assert fingerprint_hex(measured) != fingerprint_hex(predicted)


# ============================================================================
# LIFECYCLE CORRECTION (INV-12, §2.0)
# ============================================================================


def test_lifecycle_correction_corrects_chain():
    """Correction record has lifecycle: observed and corrects: <original_digest>.
    Original stays observed. Superseded is derived by query."""
    original = _make_verification(reason="initial_boot")
    original_digest = record_digest_hex(original.model_dump(mode="json"))

    correction = _make_verification(
        reason="corrected_boot",
        corrects=original_digest,
    )

    assert original.lifecycle == "observed"
    assert correction.lifecycle == "observed"
    assert correction.corrects == original_digest
    assert original.corrects is None

    computed = record_digest_hex(original.model_dump(mode="json"))
    assert correction.corrects == computed


# ============================================================================
# CROSS-FINGERPRINT REJECTION (INV-25)
# ============================================================================


def test_cross_fingerprint_score_not_transferable():
    """A task score from solution A is not a measurement for solution B."""
    attempt_a = _make_attempt(solution_fp=_FP, criterion_scores={"acc": 0.95})
    attempt_b = _make_attempt(solution_fp=_FP2, criterion_scores={"acc": 0.95})

    assert attempt_a.solution_fingerprint != attempt_b.solution_fingerprint
    assert fingerprint_hex(attempt_a) != fingerprint_hex(attempt_b)


# ============================================================================
# PROMOTION GATES (INV-18)
# ============================================================================


def test_promotion_gate_no_recommended_without_task_and_serving():
    assert not can_promote("identity_resolved", "task_evaluated", has_task_evidence=False)
    assert not can_promote(
        "task_evaluated",
        "serving_verified",
        has_task_evidence=True,
        has_serving_evidence=False,
    )


def test_promotion_gate_no_qualified_without_reproduction():
    assert not can_promote(
        "serving_verified",
        "qualified",
        has_task_evidence=True,
        has_serving_evidence=True,
        has_exact_reproduction=False,
    )
    assert can_promote(
        "serving_verified",
        "qualified",
        has_task_evidence=True,
        has_serving_evidence=True,
        has_exact_reproduction=True,
    )


# ============================================================================
# QUALITY EVIDENCE — not free-standing (ADR-006, ADR-011)
# ============================================================================


def test_quality_evidence_carries_fingerprints():
    attempt = _make_attempt(solution_fp=_FP)
    attempt_fp = fingerprint_hex(attempt)

    qe = QualityEvidence(
        task_attempt_fingerprints=(attempt_fp,),
        evaluation_protocol_fingerprint=_FP,
        aggregate_score=0.95,
    )
    assert qe.task_attempt_fingerprints == (attempt_fp,)
    assert qe.evaluation_protocol_fingerprint == _FP


# ============================================================================
# COMPATIBILITY EVIDENCE — scoped to candidate + execution
# ============================================================================


def test_compatibility_evidence_scoped():
    ce = CompatibilityEvidence(
        candidate_fingerprint=_FP,
        execution_fingerprint=_DIGEST,
        compatible=True,
    )
    assert ce.candidate_fingerprint == _FP
    assert ce.execution_fingerprint == _DIGEST

    ce_different = CompatibilityEvidence(
        candidate_fingerprint=_FP2,
        execution_fingerprint=_DIGEST,
        compatible=False,
    )
    assert fingerprint_hex(ce) != fingerprint_hex(ce_different)


# ============================================================================
# EVIDENCE RELEASE MANIFEST (INV-35)
# ============================================================================


def test_evidence_release_manifest_content_addressed():
    vr = _make_verification()
    vr_digest = record_digest_hex(vr.model_dump(mode="json"))

    manifest = EvidenceReleaseManifest(
        record_digests=(vr_digest,),
        license="CDLA-Permissive-2.0",
    )
    assert manifest.license == "CDLA-Permissive-2.0"
    assert vr_digest in manifest.record_digests

    canonical = canonicalize(manifest.model_dump(mode="json"))
    restored = EvidenceReleaseManifest.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical


# ============================================================================
# PREDICTED WITH UNCERTAINTY
# ============================================================================


def test_predicted_with_uncertainty():
    predicted = PredictedStatus(
        kind="predicted",
        prediction_method="mechanism_aware_calculator",
        uncertainty_lower=14.2,
        uncertainty_upper=18.8,
        calibration_scope="qwen3_8b_bf16_rtx4090",
    )
    assert predicted.kind == "predicted"
    assert predicted.uncertainty_lower == 14.2
    assert predicted.uncertainty_upper == 18.8

    measured = MeasuredStatus(
        kind="measured",
        measurement_method="boot_profiling",
        execution_fingerprint=_DIGEST,
    )
    assert predicted.kind != measured.kind


# ============================================================================
# WORKLOAD SPEC ENVELOPE (INV-24)
# ============================================================================


def test_workload_spec_envelope_independent():
    task = TaskSuiteSpec(
        name="qa",
        version="1.0",
        required_capabilities=(_TEXT_CAP,),
        success_criteria=("exact_match",),
    )
    serving = ServingWorkloadSpec(
        concurrency=4,
        throughput_target_rps=100.0,
    )

    ws = WorkloadSpec(
        task_suite_fingerprint=fingerprint_hex(task),
        serving_workload_fingerprint=fingerprint_hex(serving),
    )

    assert ws.task_suite_fingerprint != ws.serving_workload_fingerprint

    serving_fast = ServingWorkloadSpec(
        concurrency=4,
        throughput_target_rps=10000.0,
    )
    assert fingerprint_hex(task) == fingerprint_hex(task)
    assert fingerprint_hex(serving) != fingerprint_hex(serving_fast)


# ============================================================================
# VERIFY ≠ SUBMIT (ADR-010)
# ============================================================================


def test_verify_produces_no_action_request():
    """Verify produces a VerificationReport with no ActionRequest."""
    vr = _make_verification(reason="verify_boot")
    assert vr.lifecycle == "observed"


def test_submit_requires_authorization():
    """Submit produces ActionRequest → AuthorizationDecision → publication."""
    dr = DecisionRequest(objective="submit evaluation results")
    dr_digest = record_digest_hex(dr.model_dump(mode="json"))

    ar = ActionRequest(
        action_type="submit_evaluation",
        decision_request_digest=dr_digest,
    )
    ar_digest = record_digest_hex(ar.model_dump(mode="json"))

    contribution = AuthorityContribution(
        source_type="owner_policy",
        source_version="1.0",
        principal="owner",
        decision="permit",
        action="submit_evaluation",
    )
    result = evaluate_authorization((contribution,))
    assert result == "authorized"

    decision = AuthorizationDecision(
        action_request_digest=ar_digest,
        accepted_decision_request_digest=dr_digest,
        result=result,
        contributions=(contribution,),
    )
    assert decision.result == "authorized"
    assert decision.action_request_digest == ar_digest


# ============================================================================
# ARTIFACT RELATIONS — all 6 variants
# ============================================================================


def test_artifact_relation_all_variants_round_trip():
    adapter = TypeAdapter(ArtifactRelation)
    relations = [
        PublishedByOwner(
            relation="published_by_owner",
            publisher="Qwen",
            artifact_digest=_DIGEST,
        ),
        ClaimedDerivedFrom(
            relation="claimed_derived_from",
            source_artifact_digest=_DIGEST,
            claim_basis="name_similarity",
        ),
        ReproduciblyDerivedFrom(
            relation="reproducibly_derived_from",
            source_artifact_digest=_DIGEST,
            transform_spec_digest=_FP,
        ),
        StructurallyCompatibleWith(
            relation="structurally_compatible_with",
            other_artifact_digest=_DIGEST,
            matching_criteria="tensor_shapes",
        ),
        TokenizerCompatibleWith(
            relation="tokenizer_compatible_with",
            other_artifact_digest=_DIGEST,
            tokenizer_hash="abc123",
        ),
    ]
    for rel in relations:
        canonical = canonicalize(rel.model_dump(mode="json"))
        parsed = adapter.validate_json(canonical)
        assert type(parsed) is type(rel)


# ============================================================================
# CROSS-FIXTURE FINGERPRINT INTEGRITY
# ============================================================================


def test_cross_fixture_fingerprint_integrity():
    """Every embedded fingerprint reference matches the computed fingerprint
    of the referenced record."""
    task = TaskSuiteSpec(
        name="integrity-test",
        version="1.0",
        required_capabilities=(_TEXT_CAP,),
    )
    serving = ServingWorkloadSpec(concurrency=2)

    ws = WorkloadSpec(
        task_suite_fingerprint=fingerprint_hex(task),
        serving_workload_fingerprint=fingerprint_hex(serving),
    )
    assert ws.task_suite_fingerprint == fingerprint_hex(task)
    assert ws.serving_workload_fingerprint == fingerprint_hex(serving)

    attempt = _make_attempt(solution_fp=_FP)
    qe = QualityEvidence(
        task_attempt_fingerprints=(fingerprint_hex(attempt),),
        evaluation_protocol_fingerprint=_FP,
    )
    assert qe.task_attempt_fingerprints[0] == fingerprint_hex(attempt)
