"""Qualification verdicts computed from typed evidence records (F3).

Pure functions, no I/O.  The application layer loads records by digest and
passes the typed records here; a caller can no longer assert ``passed=True``.
Task and serving verdicts are independent (INV-24): neither implies the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apron.domain.schemas.records import TaskAttemptRecord


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str | None = None
    detail: tuple[str, ...] = ()


def final_attempts(attempts: Sequence[TaskAttemptRecord]) -> dict[str, TaskAttemptRecord]:
    """The last attempt per case: highest ``retries`` index, ties by attempt_id.

    Earlier attempts and retries stay stored and count in outcome economics;
    acceptance is judged on each case's final attempt.
    """
    final: dict[str, TaskAttemptRecord] = {}
    for attempt in attempts:
        current = final.get(attempt.case_id)
        key = (attempt.retries or 0, attempt.attempt_id)
        if current is None or key > (current.retries or 0, current.attempt_id):
            final[attempt.case_id] = attempt
    return final


def accepted_case_ids(attempts: Sequence[TaskAttemptRecord]) -> frozenset[str]:
    return frozenset(
        case for case, attempt in final_attempts(attempts).items() if attempt.accepted is True
    )


def task_verdict(
    attempts: Sequence[TaskAttemptRecord],
    *,
    solution_fingerprint: str,
    case_ids: Sequence[str],
    quality_floor: float | None,
) -> Verdict:
    """Apply the accepted request's acceptance rule to stored task attempts.

    The rule is a pass rate over the suite's cases: accepted final attempts
    divided by the number of cases.  A missing case counts as not accepted.
    With no ``quality_floor`` declared, every case must be accepted.
    """
    if not case_ids:
        return Verdict(False, "task suite declares no cases")
    foreign = sorted({a.solution_fingerprint for a in attempts} - {solution_fingerprint})
    if foreign:
        return Verdict(False, "task attempts belong to a different solution", tuple(foreign))
    accepted = accepted_case_ids(attempts) & set(case_ids)
    missing = tuple(sorted(set(case_ids) - set(final_attempts(attempts))))
    rate = len(accepted) / len(case_ids)
    floor = 1.0 if quality_floor is None else quality_floor
    detail = (f"accepted {len(accepted)}/{len(case_ids)}", *(f"missing {c}" for c in missing))
    if rate >= floor:
        return Verdict(True, None, detail)
    return Verdict(False, f"pass rate {rate:.2f} below quality floor {floor:.2f}", detail)


def reproduction_verdict(
    first: Sequence[TaskAttemptRecord],
    second: Sequence[TaskAttemptRecord],
    *,
    solution_fingerprint: str,
) -> Verdict:
    """A second task run on the same solution gives an identical accepted set."""
    for run in (first, second):
        if not run:
            return Verdict(False, "reproduction run has no attempts")
        if {a.solution_fingerprint for a in run} != {solution_fingerprint}:
            return Verdict(False, "reproduction attempts belong to a different solution")
    a, b = accepted_case_ids(first), accepted_case_ids(second)
    if a == b:
        return Verdict(True, None, (f"accepted set {sorted(a)}",))
    return Verdict(
        False,
        "accepted sets differ between runs",
        (f"first {sorted(a)}", f"second {sorted(b)}"),
    )
