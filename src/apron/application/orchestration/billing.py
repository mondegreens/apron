"""Reconcile the budget ledger with what the provider billed, pod by pod.

The ledger is what Apron tracked while it ran; the provider's bill is what
the run cost.  Compared per pod over the run window they show two kinds of
gap (L5 review, 2026-09-27):

- a pod the provider billed that no ledger entry or event names (the L0-A
  kill tests ran before the ledger existed; one pod is unattributed);
- a pod settled at the estimate because its cost could not be read back
  (a crash replay after the pod was gone).

Pure: the caller supplies the ledger, the events and the provider's rows.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from apron.application.orchestration.pods import IDLE_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

RECONCILE_PREFIX = "reconcile:unledgered:"
# Differences below this are rounding and billing granularity, not gaps.
TOLERANCE_USD = 0.01


def ledger_by_pod(ledger: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """What the ledger settled for each pod (holds are tied to pods by annotate)."""
    pod_of: dict[str, str] = {}
    settled: dict[str, float] = defaultdict(float)
    for entry in ledger:
        op, label = entry["op"], str(entry["label"])
        if op == "annotate" and entry.get("pod_id"):
            pod_of[label] = str(entry["pod_id"])
        elif op in ("settle", "spend"):
            if label.startswith(IDLE_PREFIX):
                settled[label[len(IDLE_PREFIX) :]] += float(entry["amount"])
            elif label.startswith(RECONCILE_PREFIX):
                settled[label[len(RECONCILE_PREFIX) :]] += float(entry["amount"])
            elif label in pod_of:
                settled[pod_of.pop(label)] += float(entry["amount"])
    return {pod: round(usd, 6) for pod, usd in settled.items()}


def reconcile(
    ledger: Sequence[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    billing: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-pod ledger vs billed, and the billed pods nothing in the run names."""
    billed: dict[str, float] = defaultdict(float)
    for row in billing:
        billed[str(row["podId"])] += float(row["amount"])
    tracked = ledger_by_pod(ledger)
    named = set(tracked) | {str(e["pod_id"]) for e in events if e.get("pod_id")}
    rows = []
    for pod in sorted(set(tracked) | set(billed)):
        ledger_usd, billed_usd = round(tracked.get(pod, 0.0), 6), round(billed.get(pod, 0.0), 6)
        rows.append(
            {
                "pod": pod,
                "ledger": ledger_usd,
                "billed": billed_usd if pod in billed else None,
                "difference": round(ledger_usd - billed_usd, 6) if pod in billed else None,
                "named_by_run": pod in named,
            }
        )
    unledgered = [
        r for r in rows if r["billed"] and r["pod"] not in tracked and r["billed"] > 0
    ]
    mismatched = [
        r
        for r in rows
        if r["difference"] is not None
        and r["pod"] in tracked
        and abs(r["difference"]) > TOLERANCE_USD
    ]
    not_yet_billed = [r["pod"] for r in rows if r["billed"] is None]
    return {
        "billed_total": round(sum(billed.values()), 6),
        "ledger_pod_total": round(sum(tracked.values()), 6),
        "unledgered": unledgered,
        "mismatched": mismatched,
        "not_yet_billed": not_yet_billed,
        "pods": rows,
    }
