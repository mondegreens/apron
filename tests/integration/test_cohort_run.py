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


SUITE = os.environ.get("APRON_TASK_SUITE")


def _inputs():  # type: ignore[no-untyped-def]
    """The accepted inputs; ``APRON_TASK_SUITE=v2`` for the modern models,
    ``v3`` for the deployment checks."""
    from apron.interfaces.cohort_root import TASK_SUITE_V2, TASK_SUITE_V3, load_inputs

    suites = {"v2": TASK_SUITE_V2, "v3": TASK_SUITE_V3}
    return load_inputs(task_suite=suites.get(SUITE or ""))


def _step(name: str) -> None:
    if name != STEP:
        pytest.skip(f"APRON_COHORT_STEP={STEP!r}; this is step {name!r}")


def _write(name: str, data: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / name).write_text(json.dumps(data, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------
# Capacity (PLAN risk 2): wait for Secure stock, never spin on refusals
# ---------------------------------------------------------------------------

CAPACITY_POLL_SECONDS = int(os.environ.get("APRON_CAPACITY_POLL", "120"))
CAPACITY_WAIT_SECONDS = int(os.environ.get("APRON_CAPACITY_WAIT", str(6 * 3600)))
NO_CAPACITY = "no longer any instances available"


def _log_wait(entry: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}
    with (RUN_DIR / "capacity-waits.jsonl").open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print("capacity:", entry, flush=True)


def _await_capacity(gpu: str, count: int) -> None:
    """Poll the free stock query until Secure stock exists for (gpu, count)."""
    from apron.adapters.backends.runpod import RunPodTarget

    probe = RunPodTarget(gpu_type=gpu, gpu_count=count)
    deadline = time.monotonic() + CAPACITY_WAIT_SECONDS
    waited = 0
    while True:
        status = probe.stock_status()
        if status:
            if waited:
                _log_wait({"gpu": gpu, "count": count, "stock": status, "waited_s": waited})
            return
        if time.monotonic() > deadline:
            _log_wait({"gpu": gpu, "count": count, "stock": None, "gave_up_after_s": waited})
            pytest.fail(f"no Secure stock for {count}x {gpu} after {waited}s")
        time.sleep(CAPACITY_POLL_SECONDS)
        waited += CAPACITY_POLL_SECONDS


def _when_available(gpu: str, count: int, run, attempts: int = 12):  # type: ignore[no-untyped-def]
    """Run *run* once stock exists; a refusal at creation goes back to waiting."""
    for attempt in range(attempts):
        _await_capacity(gpu, count)
        try:
            return run()
        except Exception as exc:
            if NO_CAPACITY not in str(exc):
                raise
            _log_wait({"gpu": gpu, "count": count, "refused_attempt": attempt + 1})
            time.sleep(CAPACITY_POLL_SECONDS)
    pytest.fail(f"{count}x {gpu}: refused {attempts} times despite stock")


# ---------------------------------------------------------------------------
# L0-A: the pod safety net — SIGKILL, then the next start's orphan cleanup
# ---------------------------------------------------------------------------

_PROVISION_AND_HANG = """
import os, sys, time
from pathlib import Path
from apron.adapters.backends.runpod import RunPodTarget
key = os.environ.get("RUNPOD_SSH_KEY_PATH") or RunPodTarget._detect_ssh_key()
public_key = Path(key + ".pub").read_text().strip()
t = RunPodTarget(gpu_type=sys.argv[1], leak_log=None)
t.provision(env=RunPodTarget.build_env(ssh_public_key=public_key), wait_timeout=1800)
print(t.pod_id, flush=True)
time.sleep(3600)
"""


def test_l0a_sigkill_then_orphan_cleanup() -> None:
    _step("l0a")
    from apron.adapters.backends.runpod import RunPodTarget

    gpu = os.environ.get("APRON_L0A_GPU", "NVIDIA RTX A5000")
    probe = RunPodTarget()
    before = probe.list_apron_pods()
    # The cleanup below uses max_age 0: it would terminate any apron pod.
    assert not before, f"apron pods already running, refusing to start: {before}"
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
    # APRON_L0A_MAX_AGE > 0 proves the start-of-run path (age from lastStartedAt):
    # after the image pull the pod is always older than a few minutes.
    max_age = int(os.environ.get("APRON_L0A_MAX_AGE", "0"))
    age = probe._pod_age(pod_id)
    assert age >= max_age, f"pod only {age}s old; the age path would not be exercised"
    terminated = probe.cleanup_orphaned_pods(max_age_seconds=max_age)
    after = [p["id"] for p in probe.list_apron_pods()]
    _write(
        f"l0a-kill-test-age{max_age}.json",
        {
            "gpu": gpu,
            "pod": pod_id,
            "max_age_seconds": max_age,
            "pod_age_seconds": age,
            "before": before,
            "terminated": terminated,
            "after": after,
        },
    )
    assert pod_id in terminated
    assert pod_id not in after


# ---------------------------------------------------------------------------
# L0-A3: measurement stability — Qwen3-1.7B on a 4090, booted twice
# ---------------------------------------------------------------------------


def test_l0a3_measurement_stability() -> None:
    _step("l0a3")
    from apron.application.orchestration.cohort import close_pool
    from apron.application.orchestration.scheduler import CandidateSeed
    from apron.interfaces.cohort_root import CohortPlanner, build_ports, live_rates

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
    inputs = _inputs()
    reports = []
    try:
        _l0a3_runs(planner, seed, inputs, ports, reports)
    finally:
        close_pool(ports)
    _l0a3_compare(reports)


def _l0a3_runs(planner, seed, inputs, ports, reports) -> None:  # type: ignore[no-untyped-def]
    from apron.application.orchestration.cohort import execute_solution

    for run in (1, 2):
        sp = planner.plan_seed(seed)
        sp = type(sp)(**{**sp.__dict__, "label": f"l0a3-run{run}"})
        outcome = _when_available(
            sp.requested.gpu_sku,
            sp.requested.gpu_count,
            lambda sp=sp, run=run: execute_solution(
                sp, inputs, ports, reason=f"l0a3_stability_run_{run}"
            ),
        )
        assert outcome.healthy, outcome.log_tail[-2000:]
        assert outcome.token_check and outcome.token_check["token_free"], outcome.token_check
        reports.append(ports.store.retrieve(outcome.boot_report_digest or ""))
        _write(f"f7-environ-check-run{run}.json", outcome.token_check)


def _l0a3_compare(reports) -> None:  # type: ignore[no-untyped-def]
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
    from apron.application.orchestration.cohort import close_pool
    from apron.interfaces.cohort_root import CohortPlanner, build_ports, live_rates

    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    ports = build_ports(rates=rates)
    planner = CohortPlanner(rates=rates)
    inputs = _inputs()
    only = {int(c) for c in os.environ.get("APRON_L0F_CLASSES", "1,2,3,4,5,6").split(",")}
    results: dict[int, dict] = {}
    try:
        _l0f_runs(planner, inputs, ports, only, results)
    finally:
        close_pool(ports)
    path = RUN_DIR / "l0f-results.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update({str(k): v for k, v in results.items()})  # classes run in batches
    _write("l0f-results.json", merged)


def _l0f_runs(planner, inputs, ports, only, results) -> None:  # type: ignore[no-untyped-def]
    from apron.application.orchestration.cohort import RunScope, execute_solution
    from apron.application.orchestration.remediation import SIX_CLASSES, failed_as_named

    for case in SIX_CLASSES:
        if case.failure_class not in only:
            continue
        bad = planner.plan_for(case.broken_plan, f"class{case.failure_class}-broken")
        outcome = _when_available(
            bad.requested.gpu_sku,
            bad.requested.gpu_count,
            lambda bad=bad: execute_solution(
                bad, inputs, ports, scope=RunScope(False, False), reason="fix_proof_broken_boot"
            ),
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
    # Reported, not asserted: a broken plan that does not fail as named is
    # replaced (§6.1) — that is a finding for the owner, not a test failure.


# ---------------------------------------------------------------------------
# L5: the cohort and the six fix proofs
# ---------------------------------------------------------------------------


def test_cohort_run() -> None:
    _step("cohort")
    from apron.application.orchestration.cohort import qualify_cohort, run_cohort
    from apron.application.orchestration.scheduler import (
        Coverage,
        rank_candidates,
        ranking_record,
    )
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        build_ports,
        live_rates,
        load_seed,
        staged_bytes,
        v3_evaluator,
    )

    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    planner = CohortPlanner(rates=rates, deployment_checks=SUITE == "v3")
    # A later pass names its own owner-approved list and tags its outputs
    # (cohort-ranking-<tag>.json); the first cohort's files stay as recorded.
    approved_file = os.environ.get("APRON_APPROVED", "approved-candidates.json")
    tag = f"-{os.environ['APRON_RUN_TAG']}" if os.environ.get("APRON_RUN_TAG") else ""
    approved = set(json.loads((RUN_DIR / approved_file).read_text()))
    seeds = [s for s in load_seed() if s.key in approved]
    # Weights download on the pod (no volume): wherever stock appears first,
    # the pod's own volume sized for the largest model (+15%, +10 GB).
    largest = max((staged_bytes(s.model_id) for s in seeds), default=0)
    ports = build_ports(
        rates=rates,
        evaluator=v3_evaluator(planner.plan_seed(s) for s in seeds) if SUITE == "v3" else None,
        pod_volume_gb=int(largest * 1.15 / 1e9) + 10 if largest else None,
    )
    ranking = rank_candidates(
        seeds, Coverage(), measured=[], remaining_budget=ports.budget.remaining, rates=rates
    )
    plans = [planner.plan_seed(r.seed) for r in ranking.ranked]
    inputs = _inputs()
    _write(
        f"cohort-ranking{tag}.json",
        {
            **ranking_record(
                ranking,
                candidates=seeds,
                existing=Coverage(),
                measured=[],
                remaining_budget=ports.budget.remaining,
                rates=rates,
            ),
            "authorization": inputs.authorization.model_dump(mode="json"),
        },
    )
    result = run_cohort(plans, inputs, ports)
    report = qualify_cohort(plans, inputs, ports.store, ports.clock, ports.ids)
    _write(
        f"cohort-result{tag}.json",
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
    from apron.application.orchestration.cohort import close_pool
    from apron.application.orchestration.remediation import SIX_CLASSES, prove_fix
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        build_fix_ports,
        build_ports,
        live_rates,
    )

    assert os.environ.get("ANTHROPIC_API_KEY"), "the classifier needs ANTHROPIC_API_KEY"
    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    ports = build_ports(rates=rates)
    planner = CohortPlanner(rates=rates)
    fix = build_fix_ports(planner, rates, ports.budget.record_spend)
    inputs = _inputs()
    only = {int(c) for c in os.environ.get("APRON_FIX_CLASSES", "1,2,3,4,5,6").split(",")}
    try:
        proofs = [prove_fix(c, inputs, ports, fix) for c in SIX_CLASSES if c.failure_class in only]
    finally:
        close_pool(ports)
    # A re-run of some classes replaces only those classes' entries.
    path = RUN_DIR / "fix-proofs.json"
    kept = json.loads(path.read_text()) if path.exists() else []
    merged = {p["failure_class"]: p for p in kept} | {p.failure_class: p.__dict__ for p in proofs}
    _write("fix-proofs.json", [merged[k] for k in sorted(merged)])


def test_recorded_failure_fix_proof() -> None:
    """``APRON_COHORT_STEP=recordedfix APRON_FIX_SOLUTION=<failed solution fp>
    APRON_FIX_FAMILY=<expected error family> APRON_FIX_ERROR=<text the log holds>``:
    §10.2 on a failure the cohort already recorded.  The broken plan is the
    recorded one (same solution, so its stored log is reused and nothing is
    re-booted broken), diagnosed against the rules of the vLLM it ran on;
    the corrected plan boots on the volume that holds its weights and runs
    the same task suite."""
    _step("recordedfix")
    from apron.adapters.backends.runpod_storage import RunPodStorage
    from apron.application.orchestration.cohort import close_pool
    from apron.application.orchestration.remediation import BrokenCase, prove_fix
    from apron.domain.schemas.solutions import DeploymentPlan
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        accrue_storage,
        build_fix_ports,
        build_ports,
        live_rates,
        v3_evaluator,
        weights_site,
    )

    assert os.environ.get("ANTHROPIC_API_KEY"), "the classifier needs ANTHROPIC_API_KEY"
    fp = os.environ["APRON_FIX_SOLUTION"]
    row = next(
        json.loads(line)
        for line in (RUN_DIR / "solutions.jsonl").read_text().splitlines()
        if fp in line
    )
    broken_plan = DeploymentPlan.model_validate(row["deployment_plan"])
    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    planner = CohortPlanner(rates=rates, deployment_checks=SUITE == "v3")
    broken = planner.plan_for(broken_plan, "recorded-broken")
    assert broken.solution_fp == fp, "the rebuilt plan is not the recorded solution"
    storage = RunPodStorage(os.environ["RUNPOD_API_KEY"])
    site = weights_site(storage, [broken.requested], [broken.model_id])
    assert site is not None, "no volume holds the weights"
    ports = build_ports(
        rates=rates,
        site=site,
        # The corrected plan serves the same checks: the facts are the broken
        # plan's (the correction changes max_num_seqs, not what it serves).
        evaluator=v3_evaluator([broken]) if SUITE == "v3" else None,
    )
    fix = build_fix_ports(
        planner, rates, ports.budget.record_spend, engine_version=broken.engine_version or ""
    )
    case = BrokenCase(
        failure_class=0,
        expected_family=os.environ["APRON_FIX_FAMILY"],
        broken_plan=broken_plan,
        call_site="recorded",
        expected_error=os.environ["APRON_FIX_ERROR"],
    )
    try:
        proof = prove_fix(case, _inputs(), ports, fix)
    finally:
        close_pool(ports)
        accrue_storage(site, ports.budget)
    tag = os.environ.get("APRON_RUN_TAG", fp[:16])
    _write(f"recorded-fix-{tag}.json", proof.__dict__)
    assert proof.broken_failed, proof.notes


def test_plan_variant_run() -> None:
    """``APRON_COHORT_STEP=variant APRON_BASE_SOLUTION=<recorded solution fp>
    APRON_ENGINE_SET=<json object>``: the recorded plan with the given engine
    settings changed, with the same task suite.  ``APRON_WEIGHTS=download``
    downloads the weights on the pod wherever the GPU is in stock (the only
    way since the volume was deleted, 2026-09-29); without it the run needs a
    volume that holds them.  ``APRON_GPU_SKU`` moves it to another GPU;
    ``APRON_REPEAT=1`` measures a solution that has records already.  A new
    solution (its own fingerprint and records), never a rewrite of the
    recorded one (owner, 2026-09-29: GLM-5.3-Flash without CUDA graphs after
    the engine replayed moved inputs mid-suite)."""
    _step("variant")
    from apron.adapters.backends.runpod_storage import RunPodStorage
    from apron.application.orchestration.cohort import run_cohort
    from apron.domain.schemas.solutions import DeploymentPlan
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        accrue_storage,
        build_ports,
        live_rates,
        staged_bytes,
        v3_evaluator,
        weights_site,
    )

    fp = os.environ["APRON_BASE_SOLUTION"]
    changes = json.loads(os.environ["APRON_ENGINE_SET"])
    assert isinstance(changes, dict) and changes, "APRON_ENGINE_SET is a non-empty object"
    row = next(
        json.loads(line)
        for line in (RUN_DIR / "solutions.jsonl").read_text().splitlines()
        if fp in line
    )
    base = DeploymentPlan.model_validate(row["deployment_plan"])
    engine = {**base.engine_configuration, **{k: str(v) for k, v in changes.items()}}
    alloc = dict(base.resource_allocation)
    if os.environ.get("APRON_GPU_SKU"):
        alloc["gpu_sku"] = os.environ["APRON_GPU_SKU"]
    variant = base.model_copy(
        update={"engine_configuration": engine, "resource_allocation": alloc}
    )
    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    planner = CohortPlanner(rates=rates, deployment_checks=SUITE == "v3")
    sp = planner.plan_for(variant, "variant")
    assert sp.solution_fp != fp, "the variant must be its own solution"
    evaluator = v3_evaluator([sp]) if SUITE == "v3" else None
    site = None
    if os.environ.get("APRON_WEIGHTS") == "download":
        # The pod's own volume sized for the model (+15%, +10 GB), as the cohort step.
        size = staged_bytes(sp.model_id)
        ports = build_ports(
            rates=rates, evaluator=evaluator, pod_volume_gb=int(size * 1.15 / 1e9) + 10
        )
    else:
        storage = RunPodStorage(os.environ["RUNPOD_API_KEY"])
        site = weights_site(storage, [sp.requested], [sp.model_id])
        assert site is not None, "no volume holds the weights"
        ports = build_ports(rates=rates, site=site, evaluator=evaluator)
    record: dict = {
        "base_solution": fp,
        "changes": changes,
        "gpu_sku": alloc["gpu_sku"],
        "solution": sp.solution_fp,
    }
    try:
        result = run_cohort(
            [sp],
            _inputs(),
            ports,
            repeat=os.environ.get("APRON_REPEAT") == "1",
        )
    finally:
        if site is not None:
            record["storage_cost"] = accrue_storage(site, ports.budget)
    record["result"] = {
        "executed": {k: v.__dict__ for k, v in result.executed.items()},
        "skipped": result.skipped,
        "stopped": result.stopped,
        "budget": ports.budget.summary(),
    }
    tag = os.environ.get("APRON_RUN_TAG", sp.solution_fp[:16])
    _write(f"variant-{tag}.json", record)
    assert result.stopped is None, result.stopped


# ---------------------------------------------------------------------------
# Staged weights (owner go 2026-09-27): a CPU pod downloads onto a network
# volume, then the GPU pods attach it and only load
# ---------------------------------------------------------------------------


def test_prestaged_run() -> None:
    """``APRON_COHORT_STEP=prestage APRON_APPROVED=<file> APRON_RUN_TAG=<tag>``;
    ``APRON_REPEAT=1`` measures already-measured solutions again (a repeat)."""
    _step("prestage")
    from apron.adapters.backends.runpod_storage import RunPodStorage
    from apron.application.orchestration.cohort import qualify_cohort, run_cohort
    from apron.interfaces.cohort_root import (
        CohortPlanner,
        accrue_storage,
        build_ports,
        live_rates,
        load_seed,
        stage_site,
        v3_evaluator,
        weights_site,
    )

    rates = live_rates(os.environ["RUNPOD_API_KEY"])
    # Suite v3's plans serve its checks (long context, tool calls, reasoning).
    planner = CohortPlanner(rates=rates, deployment_checks=SUITE == "v3")
    tag = f"-{os.environ['APRON_RUN_TAG']}" if os.environ.get("APRON_RUN_TAG") else "-prestage"
    approved = set(json.loads((RUN_DIR / os.environ["APRON_APPROVED"]).read_text()))
    seeds = [s for s in load_seed() if s.key in approved]
    assert seeds, "no approved seed matched"
    plans = [planner.plan_seed(s) for s in seeds]
    runnable = [p for p in plans if p.status == "planned"]
    models = sorted({p.model_id for p in runnable})
    executions = list({p.requested.model_dump_json(): p.requested for p in runnable}.values())

    storage = RunPodStorage(os.environ["RUNPOD_API_KEY"])
    site = weights_site(storage, executions, models)
    assert site is not None, f"no storage datacenter has stock for {executions}"
    ports = build_ports(
        rates=rates, site=site, evaluator=v3_evaluator(plans) if SUITE == "v3" else None
    )
    staging = stage_site(site, models, ports)
    record: dict = {
        "site": site.__dict__,
        "staging": {
            "pod_id": staging.pod_id,
            "cost": staging.cost,
            "seconds": staging.seconds,
            "error": staging.error,
            "models": [m.__dict__ for m in staging.models],
        },
    }
    _write(f"prestage{tag}.json", record)
    # Never fall back to downloading on the GPU pod: that is what this avoids.
    assert staging.ok, staging.error or [m.model_id for m in staging.models if not m.ok]

    inputs = _inputs()
    try:
        result = run_cohort(plans, inputs, ports, repeat=os.environ.get("APRON_REPEAT") == "1")
    finally:
        record["storage_cost"] = accrue_storage(site, ports.budget)
    report = qualify_cohort(plans, inputs, ports.store, ports.clock, ports.ids)
    record["result"] = {
        "executed": {k: v.__dict__ for k, v in result.executed.items()},
        "skipped": result.skipped,
        "stopped": result.stopped,
        "budget": ports.budget.summary(),
        "decision_report": report.model_dump(mode="json"),
    }
    _write(f"prestage{tag}.json", record)
    assert result.stopped is None, result.stopped
