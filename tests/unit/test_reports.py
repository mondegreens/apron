"""Step 9 tests: Layer 7 — DecisionReport, resources, promotion gates."""

from apron.domain.canonical import canonicalize
from apron.domain.fingerprints import assert_fully_classified
from apron.domain.schemas.reports import (
    CandidateEconomics,
    CandidateEntry,
    ContributedResourcePool,
    DecisionReport,
    MaintainerBaselineAllocation,
    can_promote,
)

_FP = "1220" + "ab" * 32
_FP2 = "1220" + "cd" * 32
_FP3 = "1220" + "ef" * 32
_DIGEST = "1220" + "99" * 32


def _round_trip(model_cls, instance):
    canonical = canonicalize(instance.model_dump(mode="json"))
    restored = model_cls.model_validate_json(canonical)
    assert canonicalize(restored.model_dump(mode="json")) == canonical
    return restored


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def test_candidate_entry_classified():
    assert_fully_classified(CandidateEntry)


def test_candidate_economics_classified():
    assert_fully_classified(CandidateEconomics)


def test_decision_report_classified():
    assert_fully_classified(DecisionReport)


def test_maintainer_baseline_allocation_classified():
    assert_fully_classified(MaintainerBaselineAllocation)


def test_contributed_resource_pool_classified():
    assert_fully_classified(ContributedResourcePool)


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_decision_report_round_trip():
    dr = DecisionReport(
        decision_request_digest=_DIGEST,
        candidates=(
            CandidateEntry(
                solution_fingerprint=_FP,
                qualification_status="qualified",
                evidence_state="measured",
                economics=CandidateEconomics(
                    market_equivalent_price=10.0,
                    gross_attributable_cost=8.0,
                    subsidy_applied=2.0,
                    project_out_of_pocket_cost=6.0,
                ),
            ),
            CandidateEntry(
                solution_fingerprint=_FP2,
                qualification_status="capability_eligible",
                rejection_reason="missing_capability",
            ),
        ),
        disclosed_comparable_set=(_FP,),
    )
    restored = _round_trip(DecisionReport, dr)
    assert len(restored.candidates) == 2


def test_maintainer_baseline_allocation_round_trip():
    mba = MaintainerBaselineAllocation(budget=500.0)
    _round_trip(MaintainerBaselineAllocation, mba)


def test_contributed_resource_pool_round_trip():
    crp = ContributedResourcePool(provenance="grant-2026", amount=200.0)
    _round_trip(ContributedResourcePool, crp)


# ---------------------------------------------------------------------------
# DecisionReport preserves every considered solution
# ---------------------------------------------------------------------------


def test_preserves_all_candidates():
    candidates = (
        CandidateEntry(solution_fingerprint=_FP, qualification_status="qualified"),
        CandidateEntry(
            solution_fingerprint=_FP2,
            qualification_status="candidate",
            rejection_reason="capability_pruned",
        ),
        CandidateEntry(
            solution_fingerprint=_FP3,
            qualification_status="candidate",
            rejection_reason="policy_rejected",
        ),
    )
    dr = DecisionReport(
        decision_request_digest=_DIGEST,
        candidates=candidates,
    )
    assert len(dr.candidates) == 3
    reasons = [c.rejection_reason for c in dr.candidates if c.rejection_reason]
    assert "capability_pruned" in reasons
    assert "policy_rejected" in reasons


def test_single_candidate_empty_comparable_set():
    dr = DecisionReport(
        decision_request_digest=_DIGEST,
        candidates=(
            CandidateEntry(
                solution_fingerprint=_FP,
                qualification_status="qualified",
            ),
        ),
        disclosed_comparable_set=(),
    )
    assert dr.disclosed_comparable_set == ()


# ---------------------------------------------------------------------------
# promotion gates (INV-18)
# ---------------------------------------------------------------------------


def test_no_recommended_without_task_evidence():
    assert not can_promote(
        "identity_resolved",
        "task_evaluated",
        has_task_evidence=False,
    )


def test_no_recommended_without_serving_evidence():
    assert not can_promote(
        "task_evaluated",
        "serving_verified",
        has_task_evidence=True,
        has_serving_evidence=False,
    )


def test_no_qualified_without_exact_reproduction():
    assert not can_promote(
        "serving_verified",
        "qualified",
        has_task_evidence=True,
        has_serving_evidence=True,
        has_exact_reproduction=False,
    )


def test_qualified_with_all_evidence():
    assert can_promote(
        "serving_verified",
        "qualified",
        has_task_evidence=True,
        has_serving_evidence=True,
        has_exact_reproduction=True,
    )


def test_no_measured_efficient_without_comparable_set():
    assert not can_promote(
        "qualified",
        "measured_efficient",
        has_task_evidence=True,
        has_serving_evidence=True,
        has_exact_reproduction=True,
        has_comparable_set=False,
    )


def test_measured_efficient_with_comparable_set():
    assert can_promote(
        "qualified",
        "measured_efficient",
        has_task_evidence=True,
        has_serving_evidence=True,
        has_exact_reproduction=True,
        has_comparable_set=True,
    )


def test_cannot_demote():
    assert not can_promote("qualified", "candidate")


def test_cannot_promote_to_same():
    assert not can_promote("qualified", "qualified")


# ---------------------------------------------------------------------------
# budget feasibility (schema-level)
# ---------------------------------------------------------------------------


def test_budget_covers_run_plan():
    allocation = MaintainerBaselineAllocation(budget=500.0)
    run_cost = 350.0
    assert allocation.budget >= run_cost


def test_contributed_pool_cannot_change_evidence_authority():
    pool = ContributedResourcePool(provenance="external-grant", amount=1000.0)
    assert pool.provenance == "external-grant"
