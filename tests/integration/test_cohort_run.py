"""Opt-in GPU runs for Phase 1b Part 4 (L0 proofs, the cohort, the six fix proofs).

These SPEND MONEY on RunPod Secure.  Nothing runs unless all of these hold:

- ``RUNPOD_API_KEY`` is set (and ``ANTHROPIC_API_KEY`` for fix proofs,
  ``HF_TOKEN`` for gated models);
- ``APRON_COHORT_STEP`` names the step: ``l0a``, ``l0a3``, ``l0f``, ``cohort``
  or ``fixproof``;
- the step was approved by the owner (stop points 2-4 in the run notebook).

    APRON_COHORT_STEP=l0a3 uv run pytest tests/integration/test_cohort_run.py -m cohort -s

Every record, the ledger and the events go to ``_dev_notes/cohort-run/``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

STEP = os.environ.get("APRON_COHORT_STEP", "")
RUN_DIR = Path(__file__).parents[2] / "_dev_notes" / "cohort-run"

pytestmark = [
    pytest.mark.cohort,
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("RUNPOD_API_KEY"), reason="RUNPOD_API_KEY not set"),
]


def _step(name: str) -> None:
    if name != STEP:
        pytest.skip(f"APRON_COHORT_STEP={STEP!r}; this is step {name!r}")


def _write(name: str, data: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# L0-A: the pod safety net — SIGKILL, then the next start's orphan cleanup
# ---------------------------------------------------------------------------

_PROVISION_AND_HANG = """
import sys, time
from apron.adapters.backends.runpod import RunPodTarget
t = RunPodTarget(gpu_type=sys.argv[1], leak_log=None)
t.provision(env=RunPodTarget.build_env())
print(t.pod_id, flush=True)
time.sleep(3600)
"""


def test_l0a_sigkill_then_orphan_cleanup() -> None:
    _step("l0a")
    from apron.adapters.backends.runpod import RunPodTarget

    gpu = os.environ.get("APRON_L0A_GPU", "NVIDIA RTX A5000")
    probe = RunPodTarget()
    before = probe.list_apron_pods()
    child = subprocess.Popen(
        [sys.executable, "-c", _PROVISION_AND_HANG, gpu], stdout=subprocess.PIPE, text=True
    )
    assert child.stdout is not None
    pod_id = child.stdout.readline().strip()
    assert pod_id, "child never reported a pod"
    child.send_signal(signal.SIGKILL)  # no atexit, no teardown
    child.wait()
    orphaned = [p["id"] for p in probe.list_apron_pods()]
    assert pod_id in orphaned
    time.sleep(5)
    terminated = probe.cleanup_orphaned_pods(max_age_seconds=0)
    after = [p["id"] for p in probe.list_apron_pods()]
    _write(
        "l0a-kill-test.json",
        {"gpu": gpu, "pod": pod_id, "before": before, "terminated": terminated, "after": after},
    )
    assert pod_id in terminated
    assert pod_id not in after


# ---------------------------------------------------------------------------
# L0-A3: measurement stability — Qwen3-1.7B on a 4090, booted twice
# ---------------------------------------------------------------------------


def test_l0a3_measurement_stability() -> None:
    _step("l0a3")
    from apron.application.orchestration.cohort import execute_solution
    from apron.application.orchestration.scheduler import CandidateSeed
    from apron.interfaces.cohort_root import CohortPlanner, build_ports, live_rates, load_inputs

    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    ports = build_ports(rates=rates)
    planner = CohortPlanner(rates=rates)
    seed = CandidateSeed(
        model_id="Qwen/Qwen3-1.7B",
        gpu_sku="NVIDIA GeForce RTX 4090",
        size_class="small",
        hardware_class="consumer",
        mechanism="autoregressive_decode",
        weight_gb=4.06,
    )
    inputs = load_inputs()
    reports = []
    for run in (1, 2):
        sp = planner.plan_seed(seed)
        sp = type(sp)(**{**sp.__dict__, "label": f"l0a3-run{run}"})
        outcome = execute_solution(sp, inputs, ports, reason=f"l0a3_stability_run_{run}")
        assert outcome.healthy, outcome.log_tail[-2000:]
        assert outcome.token_check and outcome.token_check["token_free"], outcome.token_check
        reports.append(ports.store.retrieve(outcome.boot_report_digest or ""))
        _write(f"f7-environ-check-run{run}.json", outcome.token_check)

    a, b = reports
    assert a and b
    weight_equal = a["model_weight_memory"] == b["model_weight_memory"]
    kv_rel = abs(a["available_kv_cache_memory"] - b["available_kv_cache_memory"]) / max(
        a["available_kv_cache_memory"], 1
    )
    act_rel = abs(a["transient_peak_headroom"] - b["transient_peak_headroom"]) / max(
        a["transient_peak_headroom"], 1
    )
    result = {
        "weight_identical": weight_equal,
        "kv_relative_difference": kv_rel,
        "activation_relative_difference": act_rel,
        "activation_uncertainty_note_required": act_rel > 0.05,
    }
    _write("l0a3-stability.json", result)
    assert weight_equal
    assert kv_rel <= 0.01


# ---------------------------------------------------------------------------
# L0-F: each broken plan fails at the named call site (step 1 of each proof)
# ---------------------------------------------------------------------------


def test_l0f_failure_reproduction() -> None:
    _step("l0f")
    from apron.application.orchestration.cohort import RunScope, execute_solution
    from apron.application.orchestration.remediation import SIX_CLASSES, failed_as_named
    from apron.interfaces.cohort_root import CohortPlanner, build_ports, live_rates, load_inputs

    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    ports = build_ports(rates=rates)
    planner = CohortPlanner(rates=rates)
    inputs = load_inputs()
    only = {int(c) for c in os.environ.get("APRON_L0F_CLASSES", "1,2,3,4,5,6").split(",")}
    results = {}
    for case in SIX_CLASSES:
        if case.failure_class not in only:
            continue
        bad = planner.plan_for(case.broken_plan, f"class{case.failure_class}-broken")
        outcome = execute_solution(
            bad, inputs, ports, scope=RunScope(False, False), reason="fix_proof_broken_boot"
        )
        results[case.failure_class] = {
            "family": case.expected_family,
            "call_site": case.call_site,
            "healthy": outcome.healthy,
            "failed_as_named": (not outcome.healthy) and failed_as_named(case, outcome.log_tail),
            "harness_failures": outcome.harness_failures,
            "failed_boot_digests": outcome.failed_boot_digests,
            "cost": outcome.cost,
        }
        (RUN_DIR / "l0f-logs").mkdir(parents=True, exist_ok=True)
        (RUN_DIR / "l0f-logs" / f"class{case.failure_class}.log").write_text(outcome.log_tail)
    _write("l0f-results.json", results)
    # Reported, not asserted: a broken plan that does not fail as named is
    # replaced (§6.1) — that is a finding for the owner, not a test failure.


# ---------------------------------------------------------------------------
# L5: the cohort and the six fix proofs
# ---------------------------------------------------------------------------


def test_cohort_run() -> None:
    _step("cohort")
    from apron.application.orchestration.cohort import qualify_cohort, run_cohort
    from apron.application.orchestration.scheduler import Coverage, rank_candidates
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        build_ports,
        live_rates,
        load_inputs,
        load_seed,
    )

    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    ports = build_ports(rates=rates)
    planner = CohortPlanner(rates=rates)
    approved = set(json.loads((RUN_DIR / "approved-candidates.json").read_text()))
    seeds = [s for s in load_seed() if s.key in approved]
    ranking = rank_candidates(
        seeds, Coverage(), measured=[], remaining_budget=ports.budget.remaining, rates=rates
    )
    plans = [planner.plan_seed(r.seed) for r in ranking.ranked]
    inputs = load_inputs()
    result = run_cohort(plans, inputs, ports)
    report = qualify_cohort(plans, inputs, ports.store, ports.clock, ports.ids)
    _write(
        "cohort-result.json",
        {
            "executed": {k: v.__dict__ for k, v in result.executed.items()},
            "prediction_errors": result.prediction_errors,
            "skipped": result.skipped,
            "stopped": result.stopped,
            "budget": ports.budget.summary(),
            "decision_report": report.model_dump(mode="json"),
        },
    )
    assert result.stopped is None, result.stopped


def test_fix_proofs() -> None:
    _step("fixproof")
    from apron.application.orchestration.remediation import SIX_CLASSES, prove_fix
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        build_fix_ports,
        build_ports,
        live_rates,
        load_inputs,
    )

    assert os.environ.get("ANTHROPIC_API_KEY"), "the classifier needs ANTHROPIC_API_KEY"
    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    ports = build_ports(rates=rates)
    planner = CohortPlanner(rates=rates)
    fix = build_fix_ports(planner, rates)
    inputs = load_inputs()
    only = {int(c) for c in os.environ.get("APRON_FIX_CLASSES", "1,2,3,4,5,6").split(",")}
    proofs = [prove_fix(c, inputs, ports, fix) for c in SIX_CLASSES if c.failure_class in only]
    _write("fix-proofs.json", [p.__dict__ for p in proofs])
