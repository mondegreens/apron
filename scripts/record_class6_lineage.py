"""Record the class 6 lineage search as a replayable fixture (PLAN §10.1 fallback).

Runs the production search for the class 6 checkpoint: the Hub lookup, the
classifier's base-model proposal (a paid call, recorded in the cohort ledger
as ``classifier:base-model:<id>``, D5) and the Hub confirmation.  Writes
``tests/fixtures/l0f/class6-lineage.json`` so
``tests/unit/test_fix_proof_real_logs.py`` replays the correction with no
network and no API call.  Keys come from the environment only.

    uv run python scripts/record_class6_lineage.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.remediation import SIX_CLASSES
from apron.domain.ports import WallClock
from apron.interfaces.cohort_root import (
    AUTHORIZED_USD,
    LINEAGE_DIR,
    REPO,
    RUN_DIR,
    lineage_search,
)

OUT = REPO / "tests" / "fixtures" / "l0f" / "class6-lineage.json"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required (paid classifier call, D5)", file=sys.stderr)
        return 1
    budget = BudgetTracker.replay(
        authorized=AUTHORIZED_USD, ledger=JsonlLedger(RUN_DIR / "ledger.jsonl"), clock=WallClock()
    )
    if budget.holds:
        print(f"open holds {sorted(budget.holds)}: a run is in progress", file=sys.stderr)
        return 1
    model_id = SIX_CLASSES[5].broken_plan.resource_allocation["model_id"]
    search = lineage_search(model_id, budget.record_spend)
    recorded = json.loads((LINEAGE_DIR / f"{model_id.replace('/', '--')}.json").read_text())
    OUT.write_text(
        json.dumps(
            {"recorded_at": datetime.now(UTC).isoformat(), **recorded},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    base = recorded["evidence"]["base_model"]
    print(f"base {base['id']!r} ({base['source']}, confirmed={base.get('confirmed')})")
    print(f"{0 if search is None else len(search.candidates)} candidates -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
