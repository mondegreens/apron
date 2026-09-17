"""Decision report generation from qualification graph output.

Preserves every candidate (even rejected ones) with rejection reason.
Single candidate → disclosed_comparable_set: [] (cannot be measured_efficient).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apron.domain.canonical import canonicalize, digest_hex
from apron.domain.schemas.reports import CandidateEntry, DecisionReport

if TYPE_CHECKING:
    from apron.domain.ports import Clock, IdGenerator
    from apron.domain.schemas.authority import DecisionRequest


def build_report(
    request: DecisionRequest,
    candidates: list[CandidateEntry],
    *,
    clock: Clock,
    id_gen: IdGenerator,
) -> DecisionReport:
    """Construct a DecisionReport from qualification graph output.

    Every candidate is preserved — rejected ones retain their rejection_reason.
    """
    request_digest = digest_hex(canonicalize(request.model_dump(mode="json")))

    comparable = tuple(
        c.solution_fingerprint for c in candidates if c.qualification_status == "qualified"
    )
    if len(comparable) < 2:
        comparable = ()

    return DecisionReport(
        decision_request_digest=request_digest,
        candidates=tuple(candidates),
        disclosed_comparable_set=comparable,
    )
