"""Qualification graph — advance candidates through evidence tiers.

Graph nodes match the schema's QualificationStatus Literal values.
All adapters injected through Protocol types — no imports from adapters/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apron.domain.schemas.reports import (
    CandidateEntry,
    QualificationStatus,
)

if TYPE_CHECKING:
    from apron.domain.ports import Clock

GRAPH_NODES: list[QualificationStatus] = [
    "candidate",
    "capability_eligible",
    "identity_resolved",
    "task_evaluated",
    "serving_verified",
    "task_reproduced",
    "qualified",
]


class QualificationGraph:
    """Advance candidates through qualification tiers based on evidence."""

    def __init__(
        self,
        *,
        clock: Clock,
    ) -> None:
        self._clock = clock

    def advance(
        self,
        candidate: Any,
        request: Any,
        *,
        capabilities_match: bool = True,
        identity_resolved: bool = True,
        task_evidence: dict[str, Any] | None = None,
        serving_evidence: dict[str, Any] | None = None,
        reproduction_evidence: dict[str, Any] | None = None,
    ) -> CandidateEntry:
        """Advance the candidate as far as evidence allows.

        Returns a CandidateEntry with qualification_status and graph_state
        recording per-node results with reasons.
        """
        from apron.domain.fingerprints import fingerprint_hex

        solution_fp = fingerprint_hex(candidate.artifact_spec)
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
            "artifact_identity": str(candidate.artifact_spec.identity.content_digest)[:16],
        }

        if task_evidence is None:
            graph_state["stopped_at"] = "identity_resolved"
            return self._entry(solution_fp, current, graph_state)

        task_passed = task_evidence.get("passed", False)
        graph_state["task_evaluated"] = {
            "passed": task_passed,
            "scores": task_evidence.get("scores", []),
        }
        if not task_passed:
            reason = task_evidence.get("reason", "task evaluation failed")
            graph_state["task_evaluated"]["reason"] = reason
            graph_state["stopped_at"] = "task_evaluated"
            return self._entry(solution_fp, current, graph_state, reason)

        current = "task_evaluated"

        if serving_evidence is None:
            graph_state["stopped_at"] = "task_evaluated"
            return self._entry(solution_fp, current, graph_state)

        serving_passed = serving_evidence.get("passed", False)
        graph_state["serving_verified"] = {
            "passed": serving_passed,
        }
        if not serving_passed:
            reason = serving_evidence.get("reason", "serving SLO failed")
            graph_state["serving_verified"]["reason"] = reason
            graph_state["stopped_at"] = "serving_verified"
            return self._entry(solution_fp, current, graph_state, reason)

        current = "serving_verified"

        if reproduction_evidence is None:
            graph_state["stopped_at"] = "serving_verified"
            return self._entry(solution_fp, current, graph_state)

        repro_passed = reproduction_evidence.get("passed", False)
        graph_state["task_reproduced"] = {"passed": repro_passed}
        if not repro_passed:
            reason = reproduction_evidence.get("reason", "reproduction failed")
            graph_state["task_reproduced"]["reason"] = reason
            graph_state["stopped_at"] = "task_reproduced"
            return self._entry(solution_fp, current, graph_state, reason)

        current = "qualified"
        return self._entry(solution_fp, current, graph_state)

    def _entry(
        self,
        solution_fp: str,
        status: QualificationStatus,
        graph_state: dict[str, Any],
        rejection_reason: str | None = None,
    ) -> CandidateEntry:
        import json

        return CandidateEntry(
            solution_fingerprint=solution_fp,
            qualification_status=status,
            qualification_graph_state=json.dumps(graph_state),
            rejection_reason=rejection_reason,
        )
