"""GPU-free cohort plan: resolve every seed candidate, plan it, rank by evidence value.

Spends nothing: Hugging Face metadata and the calculator only.  Prints the
ranked candidate list with estimated cost against the remaining budget —
the list the owner approves before the full cohort run (stop point 4).

    uv run python scripts/cohort_plan.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from apron.application.cost_estimator import PROVIDER_RATES
from apron.application.orchestration.scheduler import Coverage, rank_candidates
from apron.interfaces.cohort_root import CohortPlanner, live_rates, load_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=100.0)
    parser.add_argument("--json", dest="json_out")
    args = parser.parse_args()

    rates = live_rates(os.environ.get("RUNPOD_API_KEY")) or dict(PROVIDER_RATES["runpod"])
    source = "RunPod API" if os.environ.get("RUNPOD_API_KEY") else "offline table"
    planner = CohortPlanner(rates=rates)
    seeds = load_seed()
    planned, blocked = {}, {}
    for seed in seeds:
        try:
            planned[seed.key] = planner.plan_seed(seed)
        except Exception as exc:  # gated access, network — reported, never hidden
            blocked[seed.key] = f"{type(exc).__name__}: {exc}"[:160]

    usable = [s for s in seeds if s.key in planned]
    ranking = rank_candidates(
        usable, Coverage(), measured=[], remaining_budget=args.budget, rates=rates
    )
    rows = []
    for r in ranking.ranked:
        sp = planned[r.seed.key]
        total = sp.claim.proposed_configuration.get("total_required_bytes")
        rows.append(
            {
                "candidate": r.seed.key,
                "status": sp.status,
                "estimate_usd": r.estimated_cost,
                "predicted_total_gb": round(total / 1e9, 2) if total else None,
                "obligations": list(r.obligations),
                "solution_fingerprint": sp.solution_fp,
                "license": sp.license_observed,
                "gated": sp.gating_observed,
            }
        )
    print(f"rates: {source}; budget ${args.budget:.2f}")
    for row in rows:
        print(
            f"{row['estimate_usd']:6.2f}  {row['status']:10s} {row['candidate']:60s} "
            f"{row['predicted_total_gb']!s:>7} GB  {', '.join(row['obligations'])}"
        )
    print(f"total estimate ${sum(r['estimate_usd'] for r in rows):.2f}")
    for seed, reason in ranking.skipped:
        print(f"skipped  {seed.key}: {reason}")
    for key, reason in blocked.items():
        print(f"blocked  {key}: {reason}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"rates": source, "ranked": rows, "blocked": blocked}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
