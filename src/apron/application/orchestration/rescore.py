"""Score recorded task outputs under a new evaluation protocol, without re-running.

When the scoring rule changes (Phase 1b: a trailing full stop ignored), the
outputs the models produced are unchanged; only the verdict on them is.  Each
recorded attempt gets one new attempt under the new protocol:

- same output, token counts, time and retry index, as observed;
- ``reason="task_rescore"``, ``raw_observation_provenance`` and
  ``trace_references`` naming the attempt it re-scores;
- infrastructure cost 0 — the run that produced the output already paid.

A solution already scored under the new protocol is left alone (idempotent).
The manifest gains one entry per re-scored solution carrying the new protocol,
so the findings can say which rule scored which row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.evidence import (
    EvidenceContext,
    build_task_attempt,
    scorer_input,
    store_validated,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from apron.application.orchestration.cohort import AcceptedInputs
    from apron.application.orchestration.cohort_records import CohortRecords
    from apron.domain.ports import IdGenerator
    from apron.domain.protocols import RecordStore

RESCORE_REASON = "task_rescore"


@dataclass
class RescoreResult:
    attempts: list[str] = field(default_factory=list)  # new attempt digests
    manifest: list[dict[str, Any]] = field(default_factory=list)  # entries to append
    skipped: dict[str, str] = field(default_factory=dict)  # solution -> why


def rescore_attempts(
    records: CohortRecords,
    inputs: AcceptedInputs,
    store: RecordStore,
    score: Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]],
    ids: IdGenerator,
    now: str,
) -> RescoreResult:
    result = RescoreResult()
    by_solution: dict[str, list[tuple[str, Any]]] = {}
    for digest, attempt in sorted(records.attempts.items()):
        by_solution.setdefault(attempt.solution_fingerprint, []).append((digest, attempt))
    for sfp, attempts in sorted(by_solution.items()):
        entry = records.solutions.get(sfp)
        if entry is None:
            result.skipped[sfp] = "no manifest entry"
            continue
        ctx = EvidenceContext.bind(
            request=inputs.request_for(entry.model_id),
            task_suite=inputs.task_suite,
            application=inputs.application,
            protocol_template=inputs.protocol_template,
            solution_fp=sfp,
        )
        new_fp = ctx.evaluation_protocol_fingerprint
        if any(a.evaluation_protocol_fingerprint == new_fp for _, a in attempts):
            result.skipped[sfp] = "already scored under this protocol"
            continue
        if entry.task_suite != inputs.task_suite:
            result.skipped[sfp] = "recorded under another task suite"
            continue
        protocol = scorer_input(ctx, model_id=entry.model_id)
        recorded = [
            {
                "case_id": a.case_id,
                "output": a.output,
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "time_seconds": a.time_seconds,
            }
            for _, a in attempts
        ]
        for (digest, original), scored in zip(attempts, score(protocol, recorded), strict=True):
            attempt = build_task_attempt(
                ctx,
                scored,
                attempt_id=ids.generate(),
                retry=original.retries or 0,
                infrastructure_cost=0.0,
                trace_references=(digest,),
                failures=original.failures,
                reason=RESCORE_REASON,
                provenance=f"rescored from {digest}",
            )
            result.attempts.append(store_validated(store, attempt))
        result.manifest.append(
            entry.model_copy(update={"evaluation_protocol": ctx.protocol, "at": now}).model_dump(
                mode="json"
            )
        )
    return result
