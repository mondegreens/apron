"""Reconcile the cohort ledger with RunPod's bill for the run window.

Reads RunPod's per-pod billing from the first ledger entry's day onwards,
compares it with the ledger pod by pod, and:

- appends a ``spend`` for every pod RunPod billed that the ledger never
  tracked (label ``reconcile:unledgered:<pod>``; idempotent);
- writes ``_dev_notes/cohort-run/billing-reconciliation.json``: every pod's
  ledger and billed amounts, the mismatches, and the pods not billed yet.

Read-only against RunPod.  Re-run after billing posts (it lags by minutes).

    uv run python scripts/reconcile_billing.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.adapters.backends.runpod import RunPodTarget
from apron.application.orchestration.billing import RECONCILE_PREFIX, reconcile
from apron.application.orchestration.budget import BudgetTracker
from apron.domain.ports import WallClock
from apron.interfaces.cohort_root import AUTHORIZED_USD, RUN_DIR

OUT = RUN_DIR / "billing-reconciliation.json"


def main() -> int:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("RUNPOD_API_KEY is required", file=sys.stderr)
        return 1
    ledger_log = JsonlLedger(RUN_DIR / "ledger.jsonl")
    budget = BudgetTracker.replay(authorized=AUTHORIZED_USD, ledger=ledger_log, clock=WallClock())
    if budget.holds:
        print(f"open holds {sorted(budget.holds)}: a run is in progress", file=sys.stderr)
        return 1
    ledger = ledger_log.read_all()
    start = f"{str(ledger[0]['at'])[:10]}T00:00:00Z"
    end = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = RunPodTarget(api_key=key).billing(start=start, end=end)
    if not rows:
        print("RunPod billing returned nothing", file=sys.stderr)
        return 1
    result = reconcile(ledger, JsonlLedger(RUN_DIR / "events.jsonl").read_all(), rows)
    for row in result["unledgered"]:
        budget.record_spend(float(row["billed"]), f"{RECONCILE_PREFIX}{row['pod']}")
    result = reconcile(
        ledger_log.read_all(), JsonlLedger(RUN_DIR / "events.jsonl").read_all(), rows
    )
    OUT.write_text(
        json.dumps({"window": {"start": start, "end": end}, **result}, indent=2, sort_keys=True)
        + "\n"
    )
    print(f"billed ${result['billed_total']:.4f}; ledger pods ${result['ledger_pod_total']:.4f}")
    for row in result["mismatched"]:
        print(f"  mismatch {row['pod']}: ledger {row['ledger']:.4f} billed {row['billed']:.4f}")
    print(f"  not billed yet: {result['not_yet_billed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
