"""§7 budget ledger: hold/settle/release, crash replay, spent <= authorized."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.application.orchestration.budget import BudgetExceededError, BudgetTracker

if TYPE_CHECKING:
    from pathlib import Path


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 9, 26, tzinfo=UTC)


class _MemoryLedger:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def append(self, entry: dict) -> None:
        self.entries.append(dict(entry))

    def read_all(self) -> list[dict]:
        return [dict(e) for e in self.entries]


def _tracker(authorized: float = 100.0, ledger=None) -> BudgetTracker:  # type: ignore[no-untyped-def]
    return BudgetTracker(authorized=authorized, ledger=ledger or _MemoryLedger(), clock=_Clock())


def test_hold_settle_release() -> None:
    t = _tracker(10.0)
    t.hold(4.0, "a")
    t.hold(3.0, "b")
    assert t.remaining == pytest.approx(3.0)
    t.settle(2.5, "a")
    t.release_hold("b")
    assert t.spent == pytest.approx(2.5)
    assert t.held == 0
    assert t.summary()["remaining"] == pytest.approx(7.5)


def test_hold_beyond_remaining_is_refused() -> None:
    t = _tracker(5.0)
    t.hold(4.0, "a")
    assert not t.can_afford(2.0)
    with pytest.raises(BudgetExceededError):
        t.hold(2.0, "b")
    assert "b" not in t.holds


def test_every_operation_is_appended_before_state_changes() -> None:
    ledger = _MemoryLedger()
    t = _tracker(10.0, ledger)
    t.hold(1.0, "a")
    t.annotate_hold("a", "pod-1")
    t.settle(0.8, "a")
    t.record_spend(0.01, "classifier")
    assert [e["op"] for e in ledger.entries] == ["hold", "annotate", "settle", "spend"]


def test_overspend_is_recorded_then_stops_the_run() -> None:
    ledger = _MemoryLedger()
    t = _tracker(1.0, ledger)
    t.hold(1.0, "a")
    with pytest.raises(BudgetExceededError):
        t.settle(1.5, "a")
    assert ledger.entries[-1]["op"] == "settle"  # the money is gone; it is on the ledger


def test_overrun_of_fifty_percent_is_flagged() -> None:
    t = _tracker(100.0)
    t.hold(1.0, "a")
    t.settle(1.6, "a")
    t.hold(1.0, "b")
    t.settle(1.4, "b")
    assert t.overruns == ["a"]


def test_replay_after_simulated_crash_settles_open_hold_at_pod_cost(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    first = _tracker(20.0, ledger)
    first.hold(2.0, "done")
    first.settle(1.5, "done")
    first.hold(3.0, "crashed")
    first.annotate_hold("crashed", "pod-9")
    first.hold(1.0, "no-pod")
    # crash: process dies with two holds open

    replayed = BudgetTracker.replay(
        authorized=20.0,
        ledger=JsonlLedger(tmp_path / "ledger.jsonl"),
        clock=_Clock(),
        pod_cost=lambda pod: 2.25 if pod == "pod-9" else None,
    )
    assert replayed.holds == {}
    assert replayed.spent == pytest.approx(1.5 + 2.25 + 1.0)
    assert "crashed:replayed:provider_reported" in replayed.flags
    assert "no-pod:replayed:estimate" in replayed.flags
    again = BudgetTracker.replay(
        authorized=20.0, ledger=JsonlLedger(tmp_path / "ledger.jsonl"), clock=_Clock()
    )
    assert again.spent == pytest.approx(replayed.spent)  # replay settlements are durable


def test_torn_final_line_is_skipped_but_corruption_elsewhere_fails(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    ledger.append({"op": "hold", "label": "a", "amount": 1.0, "at": "t"})
    with path.open("a") as fh:
        fh.write('{"op": "hold", "lab')
    assert len(ledger.read_all()) == 1
    path.write_text('{"broken\n{"op": "hold", "label": "a", "amount": 1.0, "at": "t"}\n')
    with pytest.raises(ValueError, match="corrupt"):
        ledger.read_all()


def test_replay_equals_live_state(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "l.jsonl")
    live = _tracker(50.0, ledger)
    live.hold(5.0, "x")
    live.settle(4.0, "x")
    live.record_spend(0.02, "c")
    live.hold(2.0, "y")
    live.release_hold("y")
    replayed = BudgetTracker.replay(authorized=50.0, ledger=ledger, clock=_Clock())
    assert replayed.summary() == live.summary()


_ops = st.lists(
    st.tuples(
        st.sampled_from(["hold", "settle", "release", "spend"]),
        st.floats(min_value=0, max_value=30, allow_nan=False),
        st.floats(min_value=0, max_value=1, allow_nan=False),
    ),
    max_size=40,
)


@settings(max_examples=300, deadline=None)
@given(_ops)
def test_spent_never_exceeds_authorized_when_actuals_stay_within_holds(ops) -> None:  # type: ignore[no-untyped-def]
    """Property: with actual cost <= its hold, no sequence can overspend."""
    t = _tracker(100.0)
    counter = 0
    for op, amount, fraction in ops:
        if op == "hold" and t.can_afford(amount):
            counter += 1
            t.hold(amount, f"h{counter}")
        elif op == "settle" and t.holds:
            label, hold = next(iter(t.holds.items()))
            t.settle(hold.estimate * fraction, label)
        elif op == "release" and t.holds:
            t.release_hold(next(iter(t.holds)))
        elif op == "spend" and t.can_afford(amount * fraction):
            t.record_spend(amount * fraction, "s")
        assert t.spent <= t.authorized + 1e-9
        assert t.spent + t.held <= t.authorized + 1e-9
