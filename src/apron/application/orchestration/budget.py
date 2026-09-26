"""Budget ledger — hold, settle, release; crash-safe by append-before-act (§7).

Every hold, settle, release and spend is appended to the ledger *before* the
call it guards (provisioning, a classifier call).  On start the ledger is
replayed: an unsettled hold is a pod that may have run, so it becomes a
settle at the provider-reported cost for that pod, or at the estimate when
that is unavailable, and it is flagged.

The authorized total comes from the run's authorization envelope; nothing
here can widen it.  Invariant: ``spent <= authorized``.  A settle that would
break it is still recorded (the money is gone) and then raises, which stops
the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from apron.domain.ports import Clock

LedgerOp = Literal["hold", "settle", "release", "spend", "annotate"]

# Stop rule: any single run whose actual cost exceeds its estimate by 50%.
OVERRUN_FACTOR = 1.5


class LedgerLog(Protocol):
    """Append-only, durable ledger storage (a JSONL file in the cohort)."""

    def append(self, entry: dict[str, Any]) -> None: ...

    def read_all(self) -> list[dict[str, Any]]: ...


class BudgetExceededError(RuntimeError):
    """An operation would take spending above the authorized total."""


@dataclass
class _Hold:
    label: str
    estimate: float
    pod_id: str | None = None


@dataclass
class BudgetTracker:
    authorized: float
    ledger: LedgerLog
    clock: Clock
    spent: float = 0.0
    holds: dict[str, _Hold] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    overruns: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------

    @property
    def held(self) -> float:
        return sum(h.estimate for h in self.holds.values())

    @property
    def remaining(self) -> float:
        return self.authorized - self.spent - self.held

    def can_afford(self, amount: float) -> bool:
        return amount >= 0 and self.spent + self.held + amount <= self.authorized + 1e-9

    # ------------------------------------------------------------------

    def hold(self, estimate: float, label: str) -> None:
        """Reserve *estimate* before provisioning; refuse what cannot be afforded."""
        if label in self.holds:
            raise ValueError(f"hold {label!r} already open")
        if not self.can_afford(estimate):
            raise BudgetExceededError(
                f"hold {label!r} ${estimate:.2f} exceeds remaining ${self.remaining:.2f}"
            )
        self._append("hold", label, estimate)
        self.holds[label] = _Hold(label, estimate)

    def annotate_hold(self, label: str, pod_id: str) -> None:
        """Record which pod a hold pays for, so replay can ask its real cost."""
        hold = self.holds[label]
        self._append("annotate", label, 0.0, pod_id=pod_id)
        hold.pod_id = pod_id

    def hold_open_ended(self, hourly_estimate: float, label: str, pod_id: str) -> None:
        """Record a pod that could not be terminated: it may still be billing.

        Unlike ``hold`` this is never refused — it records a fact, not a plan.
        The hold stays open; the next start's ``replay`` settles it at the
        provider-reported cost for the pod and flags it.
        """
        self._append("hold", label, hourly_estimate, flag="open_ended_pod_leak")
        self._append("annotate", label, 0.0, pod_id=pod_id)
        self.holds[label] = _Hold(label, hourly_estimate, pod_id)
        self.flags.append(f"{label}:open_ended_pod_leak")

    def settle(self, actual: float, label: str, *, flag: str | None = None) -> None:
        """Convert a hold into spend at the actual cost."""
        hold = self.holds.get(label)
        if hold is None:
            raise KeyError(f"no open hold {label!r}")
        self._append("settle", label, actual, estimate=hold.estimate, flag=flag)
        del self.holds[label]
        self._spend(actual, label, hold.estimate, flag)

    def release_hold(self, label: str) -> None:
        """Drop a hold that spent nothing (M2), e.g. provisioning never happened."""
        if label not in self.holds:
            raise KeyError(f"no open hold {label!r}")
        self._append("release", label, 0.0)
        del self.holds[label]

    def record_spend(self, amount: float, label: str) -> None:
        """Spend without a hold (small metered calls such as the classifier)."""
        if not self.can_afford(amount):
            raise BudgetExceededError(
                f"spend {label!r} ${amount:.4f} exceeds remaining ${self.remaining:.4f}"
            )
        self._append("spend", label, amount)
        self._spend(amount, label, None, None)

    def summary(self) -> dict[str, Any]:
        return {
            "authorized": round(self.authorized, 6),
            "spent": round(self.spent, 6),
            "held": round(self.held, 6),
            "remaining": round(self.remaining, 6),
            "open_holds": sorted(self.holds),
            "flags": list(self.flags),
            "overruns": list(self.overruns),
        }

    # ------------------------------------------------------------------

    @classmethod
    def replay(
        cls,
        *,
        authorized: float,
        ledger: LedgerLog,
        clock: Clock,
        pod_cost: Callable[[str], float | None] | None = None,
    ) -> BudgetTracker:
        """Rebuild state from the ledger after a crash.

        Unsettled holds are settled at the provider-reported pod cost when
        ``pod_cost`` answers for the hold's pod, else at the estimate, and
        flagged ``replayed``.
        """
        tracker = cls(authorized=authorized, ledger=ledger, clock=clock)
        for entry in ledger.read_all():
            op, label, amount = entry["op"], entry["label"], float(entry["amount"])
            if op == "hold":
                tracker.holds[label] = _Hold(label, amount)
            elif op == "annotate":
                tracker.holds[label].pod_id = entry.get("pod_id")
            elif op == "settle":
                hold = tracker.holds.pop(label)
                tracker._spend(amount, label, hold.estimate, entry.get("flag"), replaying=True)
            elif op == "release":
                tracker.holds.pop(label)
            elif op == "spend":
                tracker._spend(amount, label, None, None, replaying=True)
        for label, hold in list(tracker.holds.items()):
            reported = pod_cost(hold.pod_id) if (pod_cost and hold.pod_id) else None
            source = "provider_reported" if reported is not None else "estimate"
            actual = reported if reported is not None else hold.estimate
            tracker.settle(actual, label, flag=f"replayed:{source}")
        return tracker

    # ------------------------------------------------------------------

    def _append(self, op: LedgerOp, label: str, amount: float, **extra: Any) -> None:
        entry: dict[str, Any] = {
            "op": op,
            "label": label,
            "amount": round(amount, 6),
            "at": self.clock.now().isoformat(),
        }
        entry.update({k: v for k, v in extra.items() if v is not None})
        self.ledger.append(entry)

    def _spend(
        self,
        amount: float,
        label: str,
        estimate: float | None,
        flag: str | None,
        *,
        replaying: bool = False,
    ) -> None:
        self.spent += amount
        if flag:
            self.flags.append(f"{label}:{flag}")
        if estimate is not None and estimate > 0 and amount > estimate * OVERRUN_FACTOR:
            self.overruns.append(label)
        if self.spent > self.authorized + 1e-9 and not replaying:
            raise BudgetExceededError(
                f"spent ${self.spent:.2f} exceeds authorized "
                f"${self.authorized:.2f} after {label!r}"
            )


# ---------------------------------------------------------------------------
# Cost per record (§7): phase timing and attribution in each schema's fields
# ---------------------------------------------------------------------------


@dataclass
class PhaseTiming:
    """Wall-clock marks for one pod, from the injected Clock (INV-42).

    ``provision_start`` is set *before* ``provision()`` (H11), so a pod that
    bills before it answers is still timed.
    """

    provision_start: float | None = None
    task_eval_start: float | None = None
    task_eval_end: float | None = None
    serving_start: float | None = None
    serving_end: float | None = None
    teardown_end: float | None = None

    def seconds(self) -> dict[str, float]:
        """Seconds per phase; every second from provision to teardown lands in one."""
        start, end = self.provision_start, self.teardown_end
        if start is None or end is None:
            raise ValueError("timing needs provision_start and teardown_end")
        task = _span(self.task_eval_start, self.task_eval_end)
        serving = _span(self.serving_start, self.serving_end)
        total = max(0.0, end - start)
        return {
            "provision_boot_teardown": max(0.0, total - task - serving),
            "task_evaluation": task,
            "serving": serving,
            "total": total,
        }


def _span(a: float | None, b: float | None) -> float:
    return max(0.0, b - a) if a is not None and b is not None else 0.0


@dataclass(frozen=True)
class AttributedCosts:
    boot_report: float
    per_attempt: float
    serving_report: float
    total: float


def attribute_costs(timing: PhaseTiming, hourly_rate: float, n_attempts: int) -> AttributedCosts:
    """Split one pod's cost across the records it produced.

    - the boot/memory VerificationReport carries provision, boot, idle and
      teardown time;
    - each TaskAttemptRecord carries task-evaluation seconds x rate / attempts
      (failed attempts and retries included);
    - the serving VerificationReport carries the serving window.
    The parts sum to the pod's total (local clock; the provider-reported cost
    is recorded beside it, M3).
    """
    seconds = timing.seconds()
    per_second = hourly_rate / 3600
    task_cost = seconds["task_evaluation"] * per_second
    per_attempt = task_cost / n_attempts if n_attempts else 0.0
    boot = seconds["provision_boot_teardown"] * per_second
    if not n_attempts:
        boot += task_cost  # nothing to carry it; keep the total whole
    return AttributedCosts(
        boot_report=round(boot, 6),
        per_attempt=round(per_attempt, 6),
        serving_report=round(seconds["serving"] * per_second, 6),
        total=round(seconds["total"] * per_second, 6),
    )


def verification_cost_fields(amount: float) -> dict[str, float]:
    """VerificationReport cost fields (it has no ``infrastructure_cost``).

    RunPod Secure is paid at list price with no subsidy, so market price,
    gross attributable cost and project out-of-pocket cost are equal.
    """
    return {
        "market_equivalent_price": amount,
        "gross_attributable_cost": amount,
        "project_out_of_pocket_cost": amount,
    }
