"""§7 cost per record: each schema's own fields; wrong fields fail construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from apron.application.orchestration.budget import (
    PhaseTiming,
    attribute_costs,
    verification_cost_fields,
)
from apron.domain.schemas.records import RemediationRecord, TaskAttemptRecord, VerificationReport

_FP = "1220" + "ab" * 32


def _timing() -> PhaseTiming:
    return PhaseTiming(
        provision_start=0.0,
        task_eval_start=1200.0,
        task_eval_end=1260.0,
        serving_start=1260.0,
        serving_end=1380.0,
        teardown_end=1400.0,
    )


def test_phase_seconds_cover_the_whole_pod() -> None:
    seconds = _timing().seconds()
    assert seconds == {
        "provision_boot_teardown": 1220.0,
        "task_evaluation": 60.0,
        "serving": 120.0,
        "total": 1400.0,
    }


def test_attributed_costs_sum_to_pod_total() -> None:
    costs = attribute_costs(_timing(), hourly_rate=0.74, n_attempts=3)
    total = costs.boot_report + 3 * costs.per_attempt + costs.serving_report
    assert total == pytest.approx(costs.total, abs=1e-5)
    assert costs.per_attempt == pytest.approx(60 * 0.74 / 3600 / 3, abs=1e-6)


def test_failed_boot_carries_the_whole_pod() -> None:
    timing = PhaseTiming(provision_start=0.0, teardown_end=900.0)
    costs = attribute_costs(timing, hourly_rate=0.74, n_attempts=0)
    assert costs.boot_report == pytest.approx(costs.total)
    assert costs.serving_report == 0.0


def test_timing_requires_provision_start() -> None:
    with pytest.raises(ValueError):
        PhaseTiming(teardown_end=1.0).seconds()


def test_task_attempt_uses_infrastructure_cost() -> None:
    record = TaskAttemptRecord(
        decision_fingerprint=_FP,
        task_suite_fingerprint=_FP,
        application_fingerprint=_FP,
        evaluation_protocol_fingerprint=_FP,
        solution_fingerprint=_FP,
        case_id="c",
        attempt_id="a",
        infrastructure_cost=0.004,
        market_equivalent_price=0.004,
        project_out_of_pocket_cost=0.004,
        claim_scope="task_outcome",
        production_mode=False,
        reason="r",
        lifecycle="observed",
    )
    assert record.infrastructure_cost == 0.004


def test_verification_report_uses_its_own_fields() -> None:
    report = VerificationReport.model_validate(
        {
            "target_kind": "rented-provider",
            "operator": "apron",
            "execution_fingerprint": _FP,
            "claim_scope": "boot",
            "production_mode": False,
            "reason": "r",
            "lifecycle": "observed",
            **verification_cost_fields(0.25),
        }
    )
    assert report.market_equivalent_price == 0.25


def test_verification_report_rejects_infrastructure_cost() -> None:
    with pytest.raises(ValidationError) as exc:
        VerificationReport(
            target_kind="rented-provider",
            operator="apron",
            execution_fingerprint=_FP,
            claim_scope="boot",
            production_mode=False,
            reason="r",
            lifecycle="observed",
            infrastructure_cost=0.25,  # type: ignore[call-arg]
        )
    assert exc.value.errors()[0]["loc"] == ("infrastructure_cost",)


def test_remediation_record_has_no_cost_fields() -> None:
    with pytest.raises(ValidationError):
        RemediationRecord.model_validate(
            {
                "mechanism_outcome": "verified",
                "request_outcome": "satisfied",
                "accepted_request_digest": "d",
                "task_fingerprint": _FP,
                "application_fingerprint": _FP,
                "evaluation_fingerprint": _FP,
                "corrected_plan_digest": _FP,
                "market_equivalent_price": 1.0,
            }
        )
