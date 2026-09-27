"""One live pod per requested execution for a run (PLAN §9.1 step 0 between boots).

A fresh pod per solution spent most of its time pulling the 9 GiB runner
image (5-13 min) for a boot that takes seconds to minutes (L0, 2026-09-26).
The pool keeps a pod after a solution and hands it to the next solution that
requests the same execution (provider, GPU, count, cloud, image); harness
hygiene (``prepare_boot``) cleans it between boots.

Crash safety: a pod waiting in the pool has an open ``pod-idle:<id>`` hold in
the ledger, so a crash between solutions is settled by ledger replay at the
provider-reported cost, like any other open hold.  A pod is never kept after
a harness failure (lost SSH, unclean GPU, token leak): the next solution gets
a fresh one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apron.domain.fingerprints import fingerprint_hex

if TYPE_CHECKING:
    from collections.abc import Callable

    from apron.application.orchestration.budget import BudgetTracker
    from apron.domain.ports import Clock
    from apron.domain.schemas.solutions import RequestedExecutionSpec

IDLE_PREFIX = "pod-idle:"
# Held while a pod waits in the pool; settled at the real idle time.
IDLE_HOLD_MINUTES = 10.0


@dataclass
class _Slot:
    target: Any
    requested: RequestedExecutionSpec
    idle_since: float
    idle_label: str


@dataclass
class TargetPool:
    factory: Callable[[RequestedExecutionSpec], Any]
    budget: BudgetTracker
    clock: Clock
    hourly_rate: Callable[[RequestedExecutionSpec], float]
    _slots: dict[str, _Slot] = field(default_factory=dict)
    # Pods closed while switching executions: (target, requested, error).
    closed: list[tuple[Any, RequestedExecutionSpec, BaseException | None]] = field(
        default_factory=list
    )

    @staticmethod
    def key(requested: RequestedExecutionSpec) -> str:
        return fingerprint_hex(requested)

    def acquire(self, requested: RequestedExecutionSpec) -> tuple[Any, bool]:
        """A live pod for *requested* (reused=True), or a new unprovisioned target.

        One live pod at a time: a parked pod for another execution is closed
        first, so no pod bills idle while other work runs.
        """
        slot = self._slots.pop(self.key(requested), None)
        for key in list(self._slots):
            self.closed.append(self._close_slot(self._slots.pop(key)))
        if slot is None:
            return self.factory(requested), False
        self._settle_idle(slot)
        return slot.target, True

    def keep(self, requested: RequestedExecutionSpec, target: Any) -> None:
        """Park a healthy pod for the next solution; its idle time is held."""
        label = f"{IDLE_PREFIX}{target.pod_id}"
        rate = self.hourly_rate(requested)
        self.budget.hold(round(rate * IDLE_HOLD_MINUTES / 60, 6), label)
        self.budget.annotate_hold(label, target.pod_id)
        self._slots[self.key(requested)] = _Slot(target, requested, self._now(), label)

    def has_live(self, requested: RequestedExecutionSpec) -> bool:
        return self.key(requested) in self._slots

    def live(self) -> list[Any]:
        return [slot.target for slot in self._slots.values()]

    def close(self) -> list[tuple[Any, RequestedExecutionSpec, BaseException | None]]:
        """Tear every parked pod down; settle its idle time.

        Returns (target, requested, error) per pod.
        """
        results = [self._close_slot(self._slots.pop(key)) for key in list(self._slots)]
        return results

    def take_closed(self) -> list[tuple[Any, RequestedExecutionSpec, BaseException | None]]:
        """Pods closed while switching executions since the last call."""
        closed, self.closed = self.closed, []
        return closed

    def _close_slot(self, slot: _Slot) -> tuple[Any, RequestedExecutionSpec, BaseException | None]:
        error: BaseException | None = None
        try:
            slot.target.teardown()
        except Exception as exc:  # recorded by the caller (PodLeakError included)
            error = exc
        self._settle_idle(slot)
        return slot.target, slot.requested, error

    # ------------------------------------------------------------------

    def _settle_idle(self, slot: _Slot) -> None:
        seconds = max(0.0, self._now() - slot.idle_since)
        rate = self.hourly_rate(slot.requested)
        self.budget.settle(round(seconds * rate / 3600, 6), slot.idle_label)

    def _now(self) -> float:
        return self.clock.now().timestamp()
