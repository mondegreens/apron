"""Re-score every recorded task output under the current evaluation protocol.

No GPU, no request to any model: the outputs are the ones recorded.  New
attempts are stored with ``reason="task_rescore"`` and a link to the attempt
they re-score; the manifest gains the new protocol for each solution; one
``rescored`` event per solution goes to events.jsonl.

    uv run python scripts/rescore_tasks.py
"""

from __future__ import annotations

import sys

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.adapters.backends.local_store import LocalRecordStore
from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.application.orchestration.rescore import rescore_attempts
from apron.domain.ports import UuidIdGenerator, WallClock
from apron.interfaces.cohort_root import RUN_DIR, load_cohort_run, load_inputs


def main() -> int:
    run = load_cohort_run()
    if run.records.invalid:
        print(f"invalid records: {run.records.invalid[:3]}", file=sys.stderr)
        return 1
    now = WallClock().now().isoformat()
    result = rescore_attempts(
        run.records,
        load_inputs(),
        LocalRecordStore(RUN_DIR / "records"),
        DeterministicScorer().rescore,
        UuidIdGenerator(),
        now,
    )
    manifest = JsonlLedger(RUN_DIR / "solutions.jsonl")
    events = JsonlLedger(RUN_DIR / "events.jsonl")
    for entry in result.manifest:
        manifest.append(entry)
        events.append(
            {
                "at": now,
                "event": "rescored",
                "label": entry["label"],
                "model_id": entry["model_id"],
                "solution_fingerprint": entry["solution_fingerprint"],
            }
        )
    print(f"{len(result.attempts)} attempts re-scored for {len(result.manifest)} solutions")
    for sfp, why in result.skipped.items():
        print(f"  skipped {sfp[:16]}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
