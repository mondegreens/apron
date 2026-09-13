"""Step 7 tests: Layer 5 — records, diagnosis rules, release manifests."""

from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import assert_fully_classified, fingerprint_hex
from apron.domain.schemas.records import (
    DiagnosisRule,
    EvidenceReleaseManifest,
    RemediationRecord,
    TaskAttemptRecord,
    VerificationReport,
    derive_remediation_result,
)

_FP = "1220" + "ab" * 32
_FP2 = "1220" + "cd" * 32
_DIGEST = "1220" + "99" * 32


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


def _make_task_attempt(**overrides):
    defaults = {
        "decision_fingerprint": _FP,
        "task_suite_fingerprint": _FP,
        "application_fingerprint": _FP,
        "evaluation_protocol_fingerprint": _FP,
        "solution_fingerprint": _FP,
        "case_id": "case-1",
        "attempt_id": "attempt-1",
        "claim_scope": "task_outcome",
        "production_mode": False,
        "reason": "evaluation_run",
        "lifecycle": "observed",
    }
    return TaskAttemptRecord(**(defaults | overrides))


def _make_verification_report(**overrides):
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


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_task_attempt_record_classified():
    assert_fully_classified(TaskAttemptRecord)


def test_verification_report_classified():
    assert_fully_classified(VerificationReport)


def test_remediation_record_classified():
    assert_fully_classified(RemediationRecord)


def test_diagnosis_rule_classified():
    assert_fully_classified(DiagnosisRule)


def test_evidence_release_manifest_classified():
    assert_fully_classified(EvidenceReleaseManifest)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_task_attempt_record_round_trip():
    tar = _make_task_attempt(
        criterion_scores={"accuracy": 0.95},
        accepted=True,
        turns=3,
        input_tokens=512,
        output_tokens=128,
        endpoint_cost=0.01,
    )
    _round_trip(TaskAttemptRecord, tar)


def test_verification_report_round_trip():
    vr = _make_verification_report(
        initial_total_memory=25_769_803_776,
        initial_free_memory=20_000_000_000,
        model_weight_memory=8_000_000_000,
        persistent_consumption=10_000_000_000,
        available_kv_cache_memory=6_000_000_000,
        safety_buffer=500_000_000,
    )
    _round_trip(VerificationReport, vr)


def test_remediation_record_round_trip():
    rr = RemediationRecord(
        mechanism_outcome="verified",
        request_outcome="satisfied",
        accepted_request_digest=_DIGEST,
        task_fingerprint=_FP,
        application_fingerprint=_FP,
        evaluation_fingerprint=_FP,
        corrected_plan_digest=_DIGEST,
        corrects=_DIGEST,
    )
    _round_trip(RemediationRecord, rr)


def test_diagnosis_rule_round_trip():
    dr = DiagnosisRule(
        exception_class="OutOfMemoryError",
        engine_callsite_module="vllm.worker",
        error_family="oom",
        correction_spec={"batch_size": "32"},
    )
    _round_trip(DiagnosisRule, dr)


def test_evidence_release_manifest_round_trip():
    erm = EvidenceReleaseManifest(
        record_digests=(_DIGEST, _FP),
        license="CDLA-Permissive-2.0",
    )
    _round_trip(EvidenceReleaseManifest, erm)


# ---------------------------------------------------------------------------
# TaskAttemptRecord carries exact fingerprints
# ---------------------------------------------------------------------------


def test_task_attempt_carries_exact_fingerprints():
    tar = _make_task_attempt()
    assert tar.decision_fingerprint == _FP
    assert tar.task_suite_fingerprint == _FP
    assert tar.application_fingerprint == _FP
    assert tar.evaluation_protocol_fingerprint == _FP
    assert tar.solution_fingerprint == _FP


def test_cross_fingerprint_not_measurement():
    tar_a = _make_task_attempt(solution_fingerprint=_FP)
    tar_b = _make_task_attempt(solution_fingerprint=_FP2)
    assert fingerprint_hex(tar_a) != fingerprint_hex(tar_b)


# ---------------------------------------------------------------------------
# VerificationReport non-overlapping memory fields + ExecutionTarget identity
# ---------------------------------------------------------------------------


def test_verification_report_memory_fields_non_overlapping():
    vr = _make_verification_report(
        initial_total_memory=100,
        initial_free_memory=80,
        requested_memory=60,
        model_weight_memory=30,
        persistent_consumption=40,
        transient_peak_headroom=10,
        non_pytorch_increase=5,
        cuda_graph_estimate=8,
        cuda_graph_applied=7,
        cuda_graph_actual=6,
        available_kv_cache_memory=20,
        safety_buffer=2,
        profiling_shape={"batch_size": 1, "seq_len": 1},
    )
    dumped = vr.model_dump(mode="json")
    memory_fields = [
        "initial_total_memory", "initial_free_memory", "requested_memory",
        "model_weight_memory", "persistent_consumption", "transient_peak_headroom",
        "non_pytorch_increase", "cuda_graph_estimate", "cuda_graph_applied",
        "cuda_graph_actual", "available_kv_cache_memory", "safety_buffer",
    ]
    for f in memory_fields:
        assert f in dumped
        assert dumped[f] is not None


def test_verification_report_carries_execution_target_identity():
    vr = _make_verification_report(
        target_kind="local-container",
        operator="self",
        provider="runpod",
        detected_hardware_fingerprint=_FP,
    )
    assert vr.target_kind == "local-container"
    assert vr.operator == "self"
    assert vr.provider == "runpod"
    assert vr.execution_fingerprint == _DIGEST


# ---------------------------------------------------------------------------
# RemediationRecord separate outcomes
# ---------------------------------------------------------------------------


def test_remediation_separate_outcomes():
    rr = RemediationRecord(
        mechanism_outcome="verified",
        request_outcome="violated",
        accepted_request_digest=_DIGEST,
        task_fingerprint=_FP,
        application_fingerprint=_FP,
        evaluation_fingerprint=_FP,
        corrected_plan_digest=_DIGEST,
        violated_constraints=("throughput_slo",),
    )
    assert rr.mechanism_outcome == "verified"
    assert rr.request_outcome == "violated"


def test_remediation_corrected_solution_optional():
    rr = RemediationRecord(
        mechanism_outcome="verified",
        request_outcome="satisfied",
        accepted_request_digest=_DIGEST,
        task_fingerprint=_FP,
        application_fingerprint=_FP,
        evaluation_fingerprint=_FP,
        corrected_plan_digest=_DIGEST,
        corrected_solution_digest=None,
    )
    assert rr.corrected_solution_digest is None


# ---------------------------------------------------------------------------
# derive_remediation_result
# ---------------------------------------------------------------------------


def test_derive_fixed():
    assert derive_remediation_result("verified", "satisfied") == "Fixed"


def test_derive_tradeoff():
    assert derive_remediation_result("verified", "violated") == "Alternative with trade-offs"


def test_derive_unverified():
    assert derive_remediation_result("not_evaluated", "not_evaluated") == "Unverified suggestion"
    assert derive_remediation_result("failed", "satisfied") == "Unverified suggestion"


# ---------------------------------------------------------------------------
# DiagnosisRule starts as hypothesis
# ---------------------------------------------------------------------------


def test_diagnosis_rule_starts_as_hypothesis():
    dr = DiagnosisRule(
        exception_class="OutOfMemoryError",
        engine_callsite_module="vllm.worker",
        error_family="oom",
    )
    assert dr.status == "hypothesis"
    assert dr.promoting_verification_fingerprint is None


# ---------------------------------------------------------------------------
# all records carry production_mode, reason, lifecycle
# ---------------------------------------------------------------------------


def test_all_records_carry_inv20_fields():
    tar = _make_task_attempt()
    assert hasattr(tar, "production_mode")
    assert hasattr(tar, "reason")
    assert hasattr(tar, "lifecycle")
    assert hasattr(tar, "corrects")

    vr = _make_verification_report()
    assert hasattr(vr, "production_mode")
    assert hasattr(vr, "reason")
    assert hasattr(vr, "lifecycle")
    assert hasattr(vr, "corrects")

    rr = RemediationRecord(
        mechanism_outcome="not_evaluated",
        request_outcome="not_evaluated",
        accepted_request_digest=_DIGEST,
        task_fingerprint=_FP,
        application_fingerprint=_FP,
        evaluation_fingerprint=_FP,
        corrected_plan_digest=_DIGEST,
    )
    assert hasattr(rr, "production_mode")
    assert hasattr(rr, "reason")
    assert hasattr(rr, "lifecycle")
    assert hasattr(rr, "corrects")
    assert rr.claim_scope == "remediation"
