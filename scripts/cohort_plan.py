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
    if os.environ.get("RUNPOD_API_KEY"):
        _capacity_and_pooling(ranking, planned, rates)
        _fix_proof_boots(planner, rates)
    for seed, reason in ranking.skipped:
        print(f"skipped  {seed.key}: {reason}")
    for key, reason in blocked.items():
        print(f"blocked  {key}: {reason}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"rates": source, "ranked": rows, "blocked": blocked}, fh, indent=2)
    return 0


def _capacity_and_pooling(ranking, planned, rates) -> None:  # type: ignore[no-untyped-def]
    """Which ranked rows have image-compatible Secure stock now; cost with pod reuse."""
    from apron.adapters.backends.runpod import RunPodTarget
    from apron.application.orchestration.scheduler import IMAGE_PULL_MINUTES

    print("\ncapacity now (Secure, hosts running CUDA 13.x) and cost with pod reuse:")
    groups: dict[tuple[str, int], list] = {}
    for r in ranking.ranked:
        seed = r.seed
        if seed.prediction_error:
            print(f"   GPU-free        {seed.key}")
            continue
        stock = RunPodTarget(gpu_type=seed.gpu_sku, gpu_count=seed.gpu_count).stock_status()
        print(f"   {stock or 'NO CAPACITY'!s:15s} {seed.key}  ${r.estimated_cost:.2f}")
        if stock:
            groups.setdefault((seed.gpu_sku, seed.gpu_count), []).append(r)
    total = 0.0
    for (gpu, count), rs in groups.items():
        pull = IMAGE_PULL_MINUTES / 60 * rates.get(gpu, 0.0) * count
        pooled = sum(r.estimated_cost for r in rs) - pull * (len(rs) - 1)
        total += pooled
        print(f"   pod {count}x {gpu}: {len(rs)} solutions, ${pooled:.2f} (one image pull)")
    print(f"runnable now, pooled: ${total:.2f}")


def _fix_proof_boots(planner, rates) -> None:  # type: ignore[no-untyped-def]
    """The six fixed boots: the correction from each recorded L0-F log, today's GPUs."""
    import json as _json

    from apron.adapters.backends.rule_loader import load_rules
    from apron.application.orchestration.correction import CorrectionContext
    from apron.application.orchestration.diagnosis_pipeline import run_diagnosis_pipeline
    from apron.application.orchestration.remediation import SIX_CLASSES
    from apron.interfaces.cohort_root import REPO, RULES_DIR, available_catalog, hardware_for

    rules = load_rules(RULES_DIR, "vllm", "v0.29.0")
    print("\nfix proofs — fixed boots (broken boots are the stored L0-F boots):")
    total = 0.0
    for case in SIX_CLASSES:
        fx = _json.loads(
            (REPO / "tests/fixtures/l0f" / f"class{case.failure_class}.json").read_text()
        )

        class _Replay:
            def classify(self, error, fx=fx):  # type: ignore[no-untyped-def]
                return dict(fx["classification"])

            def extract(self, error, failure_class, fx=fx):  # type: ignore[no-untyped-def]
                return dict(fx["extraction"])

        count = int(case.broken_plan.resource_allocation.get("gpu_count", "1"))
        result = run_diagnosis_pipeline(
            fx["log"],
            _Replay(),
            case.broken_plan,
            fx["model_config"],
            hardware_for(fx["gpu_sku"]),
            rules,
            correction_context=CorrectionContext(
                catalog=available_catalog(rates, count),
                predicted_total_bytes=fx["predicted_total_bytes"],
            ),
        )
        fixed = result.corrected_plan
        if fixed is None:
            print(f"   class {case.failure_class}: NO CORRECTION with today's GPUs")
            continue
        gpu = fixed.resource_allocation["gpu_sku"]
        sp = planner.plan_for(fixed, f"class{case.failure_class}-fixed")
        cost = sp.estimate  # scheduler.run_cost, as the run's hold
        rate = rates.get(gpu, 0.0) * count
        minutes = cost / rate * 60 if rate else 0.0
        total += cost
        print(f"   class {case.failure_class}: {count}x {gpu}  ~{minutes:.0f} min  ${cost:.2f}")
    print(f"fix proofs total ≈ ${total:.2f} (+ classifier calls ≈ $0.07)")


if __name__ == "__main__":
    sys.exit(main())
