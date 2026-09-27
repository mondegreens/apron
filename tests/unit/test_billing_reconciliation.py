"""Ledger vs provider bill, per pod (application/orchestration/billing.py)."""

from __future__ import annotations

import pytest

from apron.application.orchestration.billing import RECONCILE_PREFIX, ledger_by_pod, reconcile

_LEDGER = [
    {"op": "hold", "label": "run-a", "amount": 0.30},
    {"op": "annotate", "label": "run-a", "amount": 0.0, "pod_id": "podA"},
    {"op": "settle", "label": "run-a", "amount": 0.20},
    {"op": "hold", "label": "run-b", "amount": 0.48},
    {"op": "annotate", "label": "run-b", "amount": 0.0, "pod_id": "podB"},
    {"op": "settle", "label": "run-b", "amount": 0.48, "flag": "replayed:estimate"},
    {"op": "hold", "label": "pod-idle:podA", "amount": 0.1},
    {"op": "settle", "label": "pod-idle:podA", "amount": 0.01},
    {"op": "spend", "label": "classifier:x", "amount": 0.02},
]
_BILL = [
    {"podId": "podA", "amount": 0.21},
    {"podId": "podB", "amount": 0.24},
    {"podId": "podKill", "amount": 0.11},
]


def test_settlements_attach_to_the_pod_of_their_hold() -> None:
    assert ledger_by_pod(_LEDGER) == {"podA": 0.21, "podB": 0.48}


def test_unledgered_and_mismatched_pods_are_reported() -> None:
    result = reconcile(_LEDGER, [{"pod_id": "podA"}], _BILL)
    assert [r["pod"] for r in result["unledgered"]] == ["podKill"]
    assert result["unledgered"][0]["named_by_run"] is False
    assert [r["pod"] for r in result["mismatched"]] == ["podB"]  # estimate 0.48 vs 0.24
    assert result["billed_total"] == 0.56


def test_a_reconciled_spend_closes_the_gap() -> None:
    ledger = [*_LEDGER, {"op": "spend", "label": f"{RECONCILE_PREFIX}podKill", "amount": 0.11}]
    result = reconcile(ledger, [], _BILL)
    assert result["unledgered"] == []


def test_a_pod_not_billed_yet_is_listed_not_counted() -> None:
    ledger = [
        {"op": "hold", "label": "run-c", "amount": 0.5},
        {"op": "annotate", "label": "run-c", "amount": 0.0, "pod_id": "podNew"},
        {"op": "settle", "label": "run-c", "amount": 0.4},
    ]
    result = reconcile(ledger, [], [])
    assert result["not_yet_billed"] == ["podNew"]
    assert result["mismatched"] == []


def test_a_correction_closes_a_mismatch_and_replays() -> None:
    """The ledger settled a crashed pod at its estimate (0.48); the bill says
    0.24.  A signed correction closes the gap without rewriting the settle."""
    from apron.application.orchestration.billing import CORRECTION_PREFIX
    from apron.application.orchestration.budget import BudgetTracker

    ledger = [*_LEDGER, {"op": "correct", "label": f"{CORRECTION_PREFIX}podB", "amount": -0.24}]
    result = reconcile(ledger, [], _BILL)
    assert [r["pod"] for r in result["mismatched"]] == []

    class _Ledger:
        def read_all(self) -> list[dict]:
            return ledger

        def append(self, entry: dict) -> None:
            raise AssertionError("replay writes nothing")

    class _Clock:
        def now(self):  # type: ignore[no-untyped-def]
            from datetime import UTC, datetime

            return datetime(2026, 9, 27, tzinfo=UTC)

    tracker = BudgetTracker.replay(authorized=100.0, ledger=_Ledger(), clock=_Clock())
    assert tracker.spent == pytest.approx(0.20 + 0.48 + 0.01 + 0.02 - 0.24)
