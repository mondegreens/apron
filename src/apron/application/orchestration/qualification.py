"""Qualification graph — advance candidates through evidence tiers.

Graph nodes match the schema's QualificationStatus Literal values.
All adapters injected through Protocol types — no imports from adapters/.

Verdicts come from stored evidence (F3).  The caller passes record digests;
the graph loads the typed records through the migration path and applies the
acceptance rule (task), ``evaluate_serving_slos`` (serving) and an identical
accepted set on a second run (reproduction).  A missing or unloadable digest
means ``passed=False`` with a named reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.evidence import load_typed
from apron.application.orchestration.serving import evaluate_serving_slos
from apron.domain.schemas.records import TaskAttemptRecord, VerificationReport
from apron.domain.schemas.reports import (
    CandidateEntry,
    QualificationStatus,
)
from apron.domain.verdicts import Verdict, reproduction_verdict, task_verdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apron.domain.ports import Clock
    from apron.domain.protocols import RecordStore
    from apron.domain.schemas.authority import DecisionRequest
    from apron.domain.schemas.tasks import ServingWorkloadSpec, TaskSuiteSpec

GRAPH_NODES: list[QualificationStatus] = [
    "candidate",
    "capability_eligible",
    "identity_resolved",
    "task_evaluated",
    "serving_verified",
    "task_reproduced",
    "qualified",
]


@dataclass(frozen=True)
class EvidenceDigests:
    """Digests of the stored records a candidate's verdicts are computed from."""

    task_attempts: tuple[str, ...] | None = None
    serving_report: str | None = None
    reproduction_attempts: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AcceptedRequest:
    """The accepted request parts the verdicts are judged against."""

    request: DecisionRequest | None
    task_suite: TaskSuiteSpec
    serving_workload: ServingWorkloadSpec
    case_ids: tuple[str, ...] = field(default=())

    def cases(self) -> tuple[str, ...]:
        if self.case_ids:
            return self.case_ids
        return tuple(case.get("id", "") for case in self.task_suite.cases)


class QualificationGraph:
    """Advance candidates through qualification tiers based on stored evidence."""

    def __init__(
        self,
        *,
        clock: Clock,
        store: RecordStore,
    ) -> None:
        self._clock = clock
        self._store = store

    def advance(
        self,
        candidate: Any,
        accepted: AcceptedRequest,
        *,
        solution_fingerprint: str,
        evidence: EvidenceDigests | None = None,
        capabilities_match: bool = True,
        identity_resolved: bool = True,
    ) -> CandidateEntry:
        """Advance the candidate as far as stored evidence allows.

        ``solution_fingerprint`` is decided where the candidate enters
        qualification — model spec + deployment plan + requested execution
        (F4) — before deployment feasibility is checked.  Every candidate has
        one, including prediction-error candidates that never execute.
        """
        evidence = evidence or EvidenceDigests()
        solution_fp = solution_fingerprint

        artifact_spec = getattr(candidate, "artifact_spec", None)
        if artifact_spec is None:
            return self._entry(
                solution_fp,
                "candidate",
                {"stopped_at": "candidate", "reason": "no artifact spec"},
                "Missing artifact spec",
            )

        graph_state: dict[str, Any] = {}
        current: QualificationStatus = "candidate"

        if not getattr(candidate, "deployment_feasible", True):
            graph_state["stopped_at"] = "candidate"
            graph_state["reason"] = "legacy candidate — no execution fingerprint"
            return self._entry(solution_fp, current, graph_state, "Legacy entry — not deployable")

        if not capabilities_match:
            graph_state["capability_eligible"] = {"passed": False, "reason": "capability mismatch"}
            graph_state["stopped_at"] = "capability_eligible"
            return self._entry(solution_fp, current, graph_state, "Capability mismatch")

        current = "capability_eligible"
        graph_state["capability_eligible"] = {"passed": True}

        if not identity_resolved:
            graph_state["identity_resolved"] = {"passed": False, "reason": "identity not resolved"}
            graph_state["stopped_at"] = "identity_resolved"
            return self._entry(solution_fp, current, graph_state, "Identity not resolved")

        current = "identity_resolved"
        graph_state["identity_resolved"] = {
            "passed": True,
            "artifact_identity": str(artifact_spec.identity.content_digest)[:16],
        }

        if evidence.task_attempts is None:
            graph_state["stopped_at"] = "identity_resolved"
            return self._entry(solution_fp, current, graph_state)

        attempts, load_error = self._load_attempts(evidence.task_attempts)
        if load_error is not None:
            verdict = Verdict(False, load_error)
        else:
            verdict = task_verdict(
                attempts,
                solution_fingerprint=solution_fp,
                case_ids=accepted.cases(),
                quality_floor=accepted.request.quality_floor if accepted.request else None,
            )
        graph_state["task_evaluated"] = _verdict_state(verdict, evidence.task_attempts)
        if not verdict.passed:
            graph_state["stopped_at"] = "task_evaluated"
            return self._entry(solution_fp, current, graph_state, verdict.reason)

        current = "task_evaluated"

        if evidence.serving_report is None:
            graph_state["stopped_at"] = "task_evaluated"
            return self._entry(solution_fp, current, graph_state)

        verdict = self._serving_verdict(evidence.serving_report, solution_fp, accepted)
        graph_state["serving_verified"] = _verdict_state(verdict, (evidence.serving_report,))
        if not verdict.passed:
            graph_state["stopped_at"] = "serving_verified"
            return self._entry(solution_fp, current, graph_state, verdict.reason)

        current = "serving_verified"

        if evidence.reproduction_attempts is None:
            graph_state["stopped_at"] = "serving_verified"
            return self._entry(solution_fp, current, graph_state)

        second, load_error = self._load_attempts(evidence.reproduction_attempts)
        if load_error is not None:
            verdict = Verdict(False, load_error)
        else:
            verdict = reproduction_verdict(attempts, second, solution_fingerprint=solution_fp)
        graph_state["task_reproduced"] = _verdict_state(verdict, evidence.reproduction_attempts)
        if not verdict.passed:
            graph_state["stopped_at"] = "task_reproduced"
            return self._entry(solution_fp, current, graph_state, verdict.reason)

        current = "qualified"
        return self._entry(solution_fp, current, graph_state)

    # ------------------------------------------------------------------

    def _load_attempts(self, digests: Sequence[str]) -> tuple[list[TaskAttemptRecord], str | None]:
        if not digests:
            return [], "no task attempt records"
        records: list[TaskAttemptRecord] = []
        for digest in digests:
            record, reason = load_typed(self._store, digest, TaskAttemptRecord)
            if record is None:
                return [], reason
            records.append(record)
        return records, None

    def _serving_verdict(
        self, digest: str, solution_fp: str, accepted: AcceptedRequest
    ) -> Verdict:
        report, reason = load_typed(self._store, digest, VerificationReport)
        if report is None:
            return Verdict(False, reason)
        if report.solution_fingerprint != solution_fp:
            return Verdict(False, "serving report belongs to a different solution")
        result = evaluate_serving_slos(report, accepted.serving_workload)
        return Verdict(
            result.passed,
            None if result.passed else "; ".join(result.reasons),
            (result.verdict, *result.reasons),
        )

    def _entry(
        self,
        solution_fp: str,
        status: QualificationStatus,
        graph_state: dict[str, Any],
        rejection_reason: str | None = None,
    ) -> CandidateEntry:
        return CandidateEntry(
            solution_fingerprint=solution_fp,
            qualification_status=status,
            qualification_graph_state=json.dumps(graph_state),
            rejection_reason=rejection_reason,
        )


def _verdict_state(verdict: Verdict, digests: Sequence[str]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "passed": verdict.passed,
        "evidence": list(digests),
        "detail": list(verdict.detail),
    }
    if verdict.reason:
        state["reason"] = verdict.reason
    return state
