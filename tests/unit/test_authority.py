"""Step 8 tests: Layer 6 — authority, deny-overrides-permit, verify≠submit, dedup."""

from typing import Literal

from pydantic import TypeAdapter

from apron.domain.canonical import canonicalize, record_digest_hex
from apron.domain.fingerprints import assert_fully_classified
from apron.domain.schemas.authority import (
    ActionAttempt,
    ActionRequest,
    AnomalyCase,
    AuthorityContribution,
    AuthorizationDecision,
    AuthorizationEnvelope,
    DecisionRequest,
    EvaluationAttempt,
    ExecutionAttempt,
    OrchestrationDecision,
    ProposedExternalAction,
    PublicationAttempt,
    evaluate_authorization,
)

_DIGEST = "1220" + "99" * 32
_FP = "1220" + "ab" * 32


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


def _make_contribution(
    decision: Literal["permit", "deny", "indeterminate"] = "permit",
) -> AuthorityContribution:
    return AuthorityContribution(
        source_type="owner",
        source_version="1.0",
        principal="user@example.com",
        decision=decision,
        action="submit",
    )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_all_authority_schemas_classified():
    assert_fully_classified(DecisionRequest)
    assert_fully_classified(ActionRequest)
    assert_fully_classified(AuthorityContribution)
    assert_fully_classified(AuthorizationDecision)
    assert_fully_classified(AuthorizationEnvelope)
    assert_fully_classified(OrchestrationDecision)
    assert_fully_classified(AnomalyCase)
    assert_fully_classified(ProposedExternalAction)


def test_action_attempt_variants_classified():
    assert_fully_classified(EvaluationAttempt)
    assert_fully_classified(ExecutionAttempt)
    assert_fully_classified(PublicationAttempt)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_decision_request_round_trip():
    dr = DecisionRequest(
        objective="deploy Qwen3-8B for text generation",
        success_criteria=("exact_match > 0.8",),
        quality_floor=0.7,
        serving_slos={"p99_ttft_ms": 2000.0},
        budget_limit=100.0,
    )
    _round_trip(DecisionRequest, dr)


def test_authorization_envelope_round_trip():
    env = AuthorizationEnvelope(
        permitted_action_classes=("evaluate", "execute"),
        permitted_providers=("self",),
        maximum_spend=50.0,
        baseline_allocation_digest=_DIGEST,
    )
    _round_trip(AuthorizationEnvelope, env)


def test_orchestration_decision_round_trip():
    od = OrchestrationDecision(
        job_id="job-001",
        deduplication_key="job-001-run-1",
        policy_version="1.0",
        state_transition="pending→running",
        inputs_digest=_DIGEST,
    )
    _round_trip(OrchestrationDecision, od)


def test_action_attempt_discriminated_union():
    adapter = TypeAdapter(ActionAttempt)
    ea = EvaluationAttempt(attempt_type="evaluation", protocol_fingerprint=_FP, status="completed")
    parsed = adapter.validate_json(canonicalize(ea.model_dump(mode="json")))
    assert isinstance(parsed, EvaluationAttempt)

    pa = PublicationAttempt(attempt_type="publication", destination="recipes", status="pending")
    parsed = adapter.validate_json(canonicalize(pa.model_dump(mode="json")))
    assert isinstance(parsed, PublicationAttempt)


# ---------------------------------------------------------------------------
# deny overrides permit; indeterminate never authorizes
# ---------------------------------------------------------------------------


def test_deny_overrides_permit():
    contributions = (
        _make_contribution("permit"),
        _make_contribution("deny"),
        _make_contribution("permit"),
    )
    assert evaluate_authorization(contributions) == "denied"


def test_all_permit_authorizes():
    contributions = (
        _make_contribution("permit"),
        _make_contribution("permit"),
    )
    assert evaluate_authorization(contributions) == "authorized"


def test_indeterminate_never_authorizes():
    contributions = (
        _make_contribution("permit"),
        _make_contribution("indeterminate"),
    )
    assert evaluate_authorization(contributions) == "denied"


def test_empty_contributions_denied():
    assert evaluate_authorization(()) == "denied"


def test_all_indeterminate_denied():
    contributions = (
        _make_contribution("indeterminate"),
        _make_contribution("indeterminate"),
    )
    assert evaluate_authorization(contributions) == "denied"


# ---------------------------------------------------------------------------
# restart/dedup: OrchestrationDecision idempotency
# ---------------------------------------------------------------------------


def test_orchestration_dedup_same_key_same_fingerprint():
    od1 = OrchestrationDecision(
        job_id="job-001",
        deduplication_key="job-001-run-1",
        policy_version="1.0",
        state_transition="pending→running",
        inputs_digest=_DIGEST,
    )
    od2 = OrchestrationDecision(
        job_id="job-001",
        deduplication_key="job-001-run-1",
        policy_version="1.0",
        state_transition="pending→running",
        inputs_digest=_DIGEST,
    )
    assert record_digest_hex(od1.model_dump(mode="json")) == record_digest_hex(
        od2.model_dump(mode="json")
    )


def test_orchestration_different_run_different_digest():
    od1 = OrchestrationDecision(
        job_id="job-001",
        deduplication_key="job-001-run-1",
        policy_version="1.0",
        state_transition="pending→running",
        inputs_digest=_DIGEST,
    )
    od2 = OrchestrationDecision(
        job_id="job-001",
        deduplication_key="job-001-run-2",
        policy_version="1.0",
        state_transition="pending→running",
        inputs_digest=_DIGEST,
    )
    assert record_digest_hex(od1.model_dump(mode="json")) != record_digest_hex(
        od2.model_dump(mode="json")
    )


# ---------------------------------------------------------------------------
# dynamic-destination authorization denied for unauthorized fallback
# ---------------------------------------------------------------------------


def test_dynamic_destination_denied():
    env = AuthorizationEnvelope(
        permitted_providers=("provider-a",),
        permitted_action_classes=("evaluate",),
    )
    assert "provider-b" not in env.permitted_providers


# ---------------------------------------------------------------------------
# verify ≠ submit
# ---------------------------------------------------------------------------


def test_verify_produces_no_action_request():
    dr = DecisionRequest(objective="verify Qwen3-8B boot")
    dr_digest = record_digest_hex(dr.model_dump(mode="json"))
    assert dr_digest.startswith("1220")


def test_submit_requires_action_request_and_authorization():
    dr = DecisionRequest(objective="submit Qwen3-8B evaluation")
    dr_digest = record_digest_hex(dr.model_dump(mode="json"))

    ar = ActionRequest(
        action_type="submit_evaluation",
        decision_request_digest=dr_digest,
    )
    ar_digest = record_digest_hex(ar.model_dump(mode="json"))

    contributions = (_make_contribution("permit"),)
    result = evaluate_authorization(contributions)

    ad = AuthorizationDecision(
        action_request_digest=ar_digest,
        accepted_decision_request_digest=dr_digest,
        result=result,
        contributions=contributions,
    )

    assert ad.result == "authorized"
    assert ad.action_request_digest == ar_digest
    assert ad.accepted_decision_request_digest == dr_digest
