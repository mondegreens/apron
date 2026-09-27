"""Record the classifier's response to each real L0-F log (PLAN §10.3, task #32).

For each of the six broken plans: the log tail exactly as the fix proof will
read it (the stored failed-boot record), the production classification path
(``VllmEngineAdapter.classify`` / ``extract`` with the curated rules), and the
planning facts diagnosis reads (resolved config.json fields, the
calculator's predicted total).  Written to ``tests/fixtures/l0f/classN.json``
so ``tests/unit/test_fix_proof_real_logs.py`` replays them in CI with no GPU
and no API call.

Paid calls (D5): each classification's actual token cost is recorded in the
cohort ledger as ``classifier:record-l0f-classN``.  Keys come from the
environment only.

    uv run python scripts/record_l0f_classifier.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from apron.adapters.backends.ledger_file import JsonlLedger
from apron.adapters.backends.rule_loader import load_rules
from apron.adapters.backends.vllm_engine import VllmEngineAdapter
from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.diagnosis_pipeline import diagnosis_model_config
from apron.application.orchestration.remediation import SIX_CLASSES
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.ports import WallClock
from apron.interfaces.cohort_root import (
    AUTHORIZED_USD,
    REPO,
    RULES_DIR,
    RUN_DIR,
    CohortPlanner,
    live_rates,
    load_cohort_run,
)

OUT = REPO / "tests" / "fixtures" / "l0f"


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is required (paid classifier calls, D5)", file=sys.stderr)
        return 1
    run = load_cohort_run()
    rules = load_rules(RULES_DIR, "vllm", "v0.29.0")
    engine = VllmEngineAdapter(rules=rules)
    planner = CohortPlanner(rates=live_rates(os.environ.get("RUNPOD_API_KEY")))
    budget = BudgetTracker.replay(
        authorized=AUTHORIZED_USD, ledger=JsonlLedger(RUN_DIR / "ledger.jsonl"), clock=WallClock()
    )
    if budget.holds:
        print(f"open holds {sorted(budget.holds)}: a run is in progress", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for case in SIX_CLASSES:
        digest = fingerprint_hex(case.broken_plan)
        stored = [
            (d, r)
            for d, r in run.records.reports.items()
            if r.deployment_plan_digest == digest
            and "boot:model_failure" in r.failures
            and r.log_tail
        ]
        if not stored:
            print(f"class {case.failure_class}: no stored failed boot for the broken plan")
            return 1
        record_digest, report = stored[-1]
        log = report.log_tail or ""
        planned = planner.plan_for(case.broken_plan, f"class{case.failure_class}-broken")
        classification = engine.classify(log)
        full = classification.pop("_full_extraction", {})
        extraction = engine.extract(log, str(classification["failure_class"]))
        cost = float(classification.get("classifier_cost_usd") or 0.0)
        if cost > 0:
            budget.record_spend(cost, f"classifier:record-l0f-class{case.failure_class}")
        fixture = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "failure_class": case.failure_class,
            "expected_family": case.expected_family,
            "broken_plan_digest": digest,
            "source_record": record_digest,
            "log": log,
            "gpu_sku": planned.requested.gpu_sku,
            "model_config": diagnosis_model_config(dict(planned.model_config)),
            "predicted_total_bytes": planned.predicted_total_bytes,
            "classification": classification,
            "extraction": extraction,
            "classifier_usage": full.get("classifier_usage"),
        }
        path = OUT / f"class{case.failure_class}.json"
        path.write_text(json.dumps(fixture, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        print(
            f"class {case.failure_class}: diagnosed {classification['failure_class']!r} "
            f"(expected {case.expected_family!r}); extracted {extraction}; ${cost:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
