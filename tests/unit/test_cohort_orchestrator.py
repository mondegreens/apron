"""§9.1 execution protocol with a mock target (L0-C and the Layer F dry run).

Only the provider/GPU is simulated (tests/unit/_cohort_fakes.py); the
orchestrator, ledger, record building, validation and storage are real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _cohort_fakes import (
    FakeEvaluator,
    accepted_inputs,
    ports,
    solution,
)

from apron.application.orchestration.cohort import (
    classify_harness_error,
    recorded_evidence,
    run_cohort,
)
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.records import TaskAttemptRecord, VerificationReport
from apron.domain.schemas.solutions import PlanningClaim

KV_LOG = (
    "ValueError: To serve at least one request with the model's max seq len (40960), "
    "(5.62 GiB KV cache is needed, which is larger than the available KV cache memory "
    "(2.40 GiB). Based on the available memory, the estimated maximum model length is 17472.\n"
    "RuntimeError: Engine core initialization failed. See root cause above."
)


def _healthy(plan: Any, target: Any) -> str:
    return "healthy"


def _records(store: Any) -> list[dict[str, Any]]:
    return [store.retrieve(d) for d in store.search("")]


def _validated(raw: dict[str, Any]) -> Any:
    if "producer" in raw:
        return load_record(PlanningClaim, raw)
    if raw["claim_scope"] == "task_outcome":
        return load_record(TaskAttemptRecord, raw)
    return load_record(VerificationReport, raw)


# ---------------------------------------------------------------------------
# L0-C: 3 mock candidates, 1 forced failure; the ledger decrements; the loop continues
# ---------------------------------------------------------------------------


def test_l0c_three_candidates_one_forced_failure(tmp_path: Path) -> None:
    failing = "Qwen/Qwen3-8B"

    def scenario(plan: Any, target: Any) -> str:
        return KV_LOG if plan.resource_allocation["model_id"] == failing else "healthy"

    cohort_ports, _, targets, events = ports(tmp_path, scenario)
    plans = [
        solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090"),
        solution(
            failing, "NVIDIA GeForce RTX 4090", engine_configuration={"max_model_len": "40960"}
        ),
        solution("mistralai/Mistral-7B-Instruct-v0.3", "NVIDIA L4"),
    ]
    before = cohort_ports.budget.remaining
    result = run_cohort(plans, accepted_inputs(), cohort_ports)

    assert set(result.executed) == {p.label for p in plans}
    healthy = [o for o in result.executed.values() if o.healthy]
    failed = [o for o in result.executed.values() if not o.healthy]
    assert len(healthy) == 2 and len(failed) == 1
    assert failed[0].failed_boot_digests and "estimated maximum model length" in failed[0].log_tail
    assert all(t.torn_down for t in targets)
    assert cohort_ports.budget.remaining < before
    assert cohort_ports.budget.holds == {}
    assert cohort_ports.budget.spent == pytest.approx(
        sum(o.cost for o in result.executed.values())
    )
    assert [e["event"] for e in events.entries].count("executed") == 3


def test_dry_run_records_are_schema_valid_with_distinct_solution_fps(tmp_path: Path) -> None:
    """Layer F gate: attempts, verification and (below) remediation records validate."""
    cohort_ports, *_ = ports(tmp_path, _healthy)
    plans = [
        solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090"),
        solution("Qwen/Qwen3-1.7B", "NVIDIA L4"),
        solution("Qwen/Qwen3-8B", "NVIDIA RTX A6000"),
    ]
    run_cohort(plans, accepted_inputs(), cohort_ports)
    records = [_validated(r) for r in _records(cohort_ports.store)]
    attempts = [r for r in records if isinstance(r, TaskAttemptRecord)]
    reports = [r for r in records if isinstance(r, VerificationReport)]
    assert len(attempts) == 3 * 3
    assert {r.claim_scope for r in reports} == {"memory", "serving_performance"}
    assert len({p.solution_fp for p in plans}) == 3
    assert {a.solution_fingerprint for a in attempts} == {p.solution_fp for p in plans}
    assert all(a.infrastructure_cost is not None for a in attempts)
    assert all(r.market_equivalent_price is not None for r in reports)
    assert all(r.operator == "apron" for r in reports)
    placeholder = "1220" + "00" * 32
    assert not any(placeholder in str(r.model_dump()) for r in records)


def test_memory_report_carries_prediction_delta_and_plan_digest(tmp_path: Path) -> None:
    from apron.domain.fingerprints import fingerprint_hex

    cohort_ports, *_ = ports(tmp_path, _healthy)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    outcome = run_cohort([sp], accepted_inputs(), cohort_ports).executed[sp.label]
    raw = cohort_ports.store.retrieve(outcome.boot_report_digest or "")
    report = load_record(VerificationReport, raw or {})
    assert report.deployment_plan_digest == fingerprint_hex(sp.plan)
    assert report.predicted_minus_measured is not None
    assert set(report.predicted_minus_measured) >= {"weight_memory", "total"}
    assert report.phase_seconds is not None and report.phase_seconds["total"] > 0
    assert report.provider_reported_cost == 0.123


def test_serving_report_has_computed_slo_verdict(tmp_path: Path) -> None:
    cohort_ports, *_ = ports(tmp_path, _healthy)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    outcome = run_cohort([sp], accepted_inputs(), cohort_ports).executed[sp.label]
    report = load_record(
        VerificationReport, cohort_ports.store.retrieve(outcome.serving_digest or "") or {}
    )
    assert report.serving_slo_verdict == "pass"
    assert report.serving_latency_ms and report.serving_latency_ms["p99_ttft_ms"] == 850.0


def test_prediction_error_candidates_spend_no_gpu(tmp_path: Path) -> None:
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy)
    mamba = solution("state-spaces/mamba-2.8b-hf", "NVIDIA GeForce RTX 4090")
    infeasible = solution("Qwen/Qwen3-32B", "NVIDIA GeForce RTX 4090")
    assert (mamba.status, infeasible.status) == ("unknown", "infeasible")
    result = run_cohort([mamba, infeasible], accepted_inputs(), cohort_ports)
    assert targets == [] and engine.booted == []
    assert cohort_ports.budget.spent == 0
    for label, status in ((mamba.label, "unknown"), (infeasible.label, "infeasible")):
        claim = load_record(
            PlanningClaim, cohort_ports.store.retrieve(result.prediction_errors[label]) or {}
        )
        assert claim.proposed_configuration["status"] == status
        assert claim.solution_fingerprint is not None


def test_harness_error_is_stored_retried_once_and_not_diagnosed(tmp_path: Path) -> None:
    calls = {"n": 0}

    def scenario(plan: Any, target: Any) -> Any:
        calls["n"] += 1
        return (
            ("harness", "OSError: [Errno 98] Address already in use")
            if calls["n"] == 1
            else "healthy"
        )

    cohort_ports, *_ = ports(tmp_path, scenario)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    outcome = run_cohort([sp], accepted_inputs(), cohort_ports).executed[sp.label]
    assert outcome.healthy
    assert outcome.harness_failures == ["harness:port_in_use"]
    failed = load_record(
        VerificationReport, cohort_ports.store.retrieve(outcome.failed_boot_digests[0]) or {}
    )
    assert failed.failures == ("harness:port_in_use",)
    assert failed.boot_outcome == "failed"
    # exit gate 4: failed attempts and retries carry cost in their own fields
    assert failed.market_equivalent_price is not None and failed.market_equivalent_price > 0
    healthy = load_record(
        VerificationReport, cohort_ports.store.retrieve(outcome.boot_report_digest or "") or {}
    )
    assert healthy.market_equivalent_price == failed.market_equivalent_price
    stored = [_validated(r) for r in _records(cohort_ports.store)]
    per_record = sum(
        (r.infrastructure_cost or 0.0)
        if isinstance(r, TaskAttemptRecord)
        else (r.market_equivalent_price or 0.0)
        for r in stored
        if not isinstance(r, PlanningClaim)
    )
    assert per_record == pytest.approx(outcome.cost, abs=1e-4)


@pytest.mark.parametrize(
    ("log", "kind"),
    [
        ("OSError: [Errno 98] Address already in use", "harness:port_in_use"),
        ("OSError: [Errno 28] No space left on device", "harness:disk_full"),
        ("huggingface_hub.errors.GatedRepoError: 401 Client Error", "harness:hf_auth"),
        ("RepositoryNotFoundError: 404 Client Error", "harness:hf_not_found"),
        ("Temporary failure in name resolution", "harness:network"),
        (KV_LOG, None),
    ],
)
def test_harness_classification(log: str, kind: str | None) -> None:
    assert classify_harness_error(log) == kind


def test_transport_failures_are_retried_and_both_attempts_stored(tmp_path: Path) -> None:
    cohort_ports, *_ = ports(
        tmp_path, _healthy, evaluator=FakeEvaluator(transport_fail_once=frozenset({"fact-1"}))
    )
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    outcome = run_cohort([sp], accepted_inputs(), cohort_ports).executed[sp.label]
    attempts = [
        load_record(TaskAttemptRecord, cohort_ports.store.retrieve(d) or {})
        for d in outcome.attempt_digests
    ]
    fact = sorted((a for a in attempts if a.case_id == "fact-1"), key=lambda a: a.retries or 0)
    assert [a.retries for a in fact] == [0, 1]
    assert fact[0].failures == ("evaluation:ReadTimeout",) and fact[1].accepted is True


def test_resume_skips_complete_and_reruns_only_missing_steps(tmp_path: Path) -> None:
    cohort_ports, engine, *_ = ports(tmp_path, _healthy)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    run_cohort([sp], accepted_inputs(), cohort_ports)
    again = run_cohort([sp], accepted_inputs(), cohort_ports)
    assert again.skipped == {sp.label: "already measured"}
    assert len(engine.booted) == 1

    # drop the serving report: resume re-runs serving only, earlier records stay
    recorded = recorded_evidence(cohort_ports.store, sp.solution_fp)
    serving_path = next((tmp_path / "records").rglob(f"{recorded.serving_reports[0]}.json"))
    serving_path.unlink()
    attempts_before = len(recorded.attempts)
    third = run_cohort([sp], accepted_inputs(), cohort_ports)
    after = recorded_evidence(cohort_ports.store, sp.solution_fp)
    assert sp.label in third.executed
    assert len(after.attempts) == attempts_before  # task evaluation not re-run
    assert len(after.serving_reports) == 1


def test_outside_authorization_is_refused(tmp_path: Path) -> None:
    cohort_ports, engine, *_ = ports(tmp_path, _healthy)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    result = run_cohort(
        [sp], accepted_inputs(hard_target_constraints={"cloud_type": "COMMUNITY"}), cohort_ports
    )
    assert "PermissionError" in result.skipped[sp.label]
    assert engine.booted == []


def test_budget_it_cannot_afford_is_skipped(tmp_path: Path) -> None:
    cohort_ports, engine, *_ = ports(tmp_path, _healthy, authorized=0.01)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    result = run_cohort([sp], accepted_inputs(), cohort_ports)
    assert "exceeds remaining budget" in result.skipped[sp.label]
    assert engine.booted == []


def test_ledger_replay_equals_stored_costs(tmp_path: Path) -> None:
    from apron.application.orchestration.budget import BudgetTracker

    cohort_ports, *_ = ports(tmp_path, _healthy)
    plans = [
        solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090"),
        solution("Qwen/Qwen3-8B", "NVIDIA RTX A6000"),
    ]
    result = run_cohort(plans, accepted_inputs(), cohort_ports)
    replayed = BudgetTracker.replay(
        authorized=100.0, ledger=cohort_ports.budget.ledger, clock=cohort_ports.clock
    )
    stored = sum(o.cost for o in result.executed.values())
    assert replayed.spent == pytest.approx(stored)
    records = [_validated(r) for r in _records(cohort_ports.store)]
    per_record = sum(
        (r.infrastructure_cost or 0.0)
        if isinstance(r, TaskAttemptRecord)
        else (r.market_equivalent_price or 0.0)
        for r in records
        if not isinstance(r, PlanningClaim)
    )
    assert per_record == pytest.approx(stored, abs=1e-4)


def test_orchestrator_has_no_adapter_imports() -> None:
    """R-4: the application modules never import apron.adapters."""
    import ast

    import apron.application.orchestration as pkg

    root = Path(pkg.__file__).parent
    for name in ("cohort", "budget", "scheduler", "serving", "remediation", "evidence"):
        tree = ast.parse((root / f"{name}.py").read_text())
        imported = [
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ] + [a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names]
        assert not [m for m in imported if m.startswith("apron.adapters")], name


def test_qualification_runs_on_stored_evidence(tmp_path: Path) -> None:
    """F4: the production caller of QualificationGraph.advance, with the solution
    fingerprint decided at planning — prediction-error candidates included."""
    from apron.application.orchestration.cohort import qualify_cohort

    cohort_ports, *_ = ports(tmp_path, _healthy)
    measured = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    mamba = solution("state-spaces/mamba-2.8b-hf", "NVIDIA GeForce RTX 4090")
    inputs = accepted_inputs()
    run_cohort([measured, mamba], inputs, cohort_ports)
    report = qualify_cohort(
        [measured, mamba], inputs, cohort_ports.store, cohort_ports.clock, cohort_ports.ids
    )
    by_fp = {c.solution_fingerprint: c for c in report.candidates}
    assert by_fp[measured.solution_fp].qualification_status == "serving_verified"
    assert by_fp[mamba.solution_fp].qualification_status == "candidate"
    assert "not deployable" in (by_fp[mamba.solution_fp].rejection_reason or "")


def test_every_attempt_digest_reproduces_from_validated_objects(tmp_path: Path) -> None:
    """§8 / exit gate 3: each fingerprint equals fingerprint_hex(<validated object>)."""
    from apron.application.orchestration.evidence import EvidenceContext, solution_identity
    from apron.domain.fingerprints import fingerprint_hex

    cohort_ports, *_ = ports(tmp_path, _healthy)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    inputs = accepted_inputs()
    outcome = run_cohort([sp], inputs, cohort_ports).executed[sp.label]
    ctx = EvidenceContext.bind(
        request=inputs.request_for(sp.model_id),
        task_suite=inputs.task_suite,
        application=inputs.application,
        protocol_template=inputs.protocol_template,
        solution_fp=sp.solution_fp,
    )
    for digest in outcome.attempt_digests:
        attempt = load_record(TaskAttemptRecord, cohort_ports.store.retrieve(digest) or {})
        assert attempt.decision_fingerprint == fingerprint_hex(inputs.request_for(sp.model_id))
        assert attempt.task_suite_fingerprint == fingerprint_hex(inputs.task_suite)
        assert attempt.application_fingerprint == fingerprint_hex(inputs.application)
        assert attempt.evaluation_protocol_fingerprint == fingerprint_hex(ctx.protocol)
        assert attempt.solution_fingerprint == fingerprint_hex(
            solution_identity(sp.model_spec, sp.plan, sp.requested)
        )


def test_candidate_model_is_in_scope_of_its_decision_request() -> None:
    inputs = accepted_inputs()
    a, b = inputs.request_for("Qwen/Qwen3-1.7B"), inputs.request_for("Qwen/Qwen3-8B")
    assert a.permitted_resources == ("Qwen/Qwen3-1.7B",)
    assert a != b


def test_token_in_vllm_environ_stops_the_run(tmp_path: Path) -> None:
    """INV-13 / F7: a token in the vLLM process environment is not recorded and
    carried on — the solution is not measured and the whole run stops."""
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy)
    object.__setattr__(engine, "_token_in_environ", True)
    first = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    second = solution("Qwen/Qwen3-1.7B", "NVIDIA L4")
    result = run_cohort([first, second], accepted_inputs(), cohort_ports)
    assert result.stopped and "TokenLeakError" in result.stopped
    assert len(targets) == 1 and targets[0].torn_down  # second candidate never provisioned
    recorded = recorded_evidence(cohort_ports.store, first.solution_fp)
    assert (
        recorded.memory_reports == []
        and recorded.attempts == []
        and recorded.serving_reports == []
    )
    leak = load_record(
        VerificationReport, cohort_ports.store.retrieve(recorded.failed_boots[0]) or {}
    )
    assert leak.failures == ("harness:token_in_vllm_environ",)
    assert leak.market_equivalent_price is not None
    assert cohort_ports.budget.holds == {}


def _cost_on_records(store: Any) -> float:
    stored = [_validated(r) for r in _records(store)]
    return sum(
        (r.infrastructure_cost or 0.0)
        if isinstance(r, TaskAttemptRecord)
        else (r.market_equivalent_price or 0.0)
        for r in stored
        if not isinstance(r, PlanningClaim)
    )


def test_runner_image_without_f7_stops_the_run_settled_and_recorded(tmp_path: Path) -> None:
    """(a) A run-level harness condition: one pod is paid for, settled, recorded;
    the run stops before the second candidate is provisioned."""
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy)
    engine.runner_supports_token_isolation = lambda target: False  # type: ignore[method-assign]
    first = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    second = solution("Qwen/Qwen3-1.7B", "NVIDIA L4")
    result = run_cohort([first, second], accepted_inputs(), cohort_ports)
    assert result.stopped and "RunnerImageError" in result.stopped
    assert len(targets) == 1 and targets[0].torn_down
    assert cohort_ports.budget.holds == {}
    assert cohort_ports.budget.spent > 0
    assert _cost_on_records(cohort_ports.store) == pytest.approx(
        cohort_ports.budget.spent, abs=1e-4
    )
    failed = recorded_evidence(cohort_ports.store, first.solution_fp).failed_boots
    report = load_record(VerificationReport, cohort_ports.store.retrieve(failed[0]) or {})
    assert report.failures == ("harness:runner_image_lacks_f7",)


def test_benchmark_error_keeps_paid_measurements_and_settles(tmp_path: Path) -> None:
    """(b) An error after a healthy boot: the memory report and task attempts
    already measured are stored with their costs; the hold is settled."""
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy)

    def broken_bench(target: Any, **_: Any) -> dict[str, Any]:
        raise RuntimeError("vllm bench serve produced no result: ")

    engine.benchmark_serving = broken_bench  # type: ignore[method-assign]
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    result = run_cohort([sp], accepted_inputs(), cohort_ports)
    assert "RuntimeError" in result.skipped[sp.label]
    assert targets[0].torn_down
    assert cohort_ports.budget.holds == {}
    recorded = recorded_evidence(cohort_ports.store, sp.solution_fp)
    assert len(recorded.memory_reports) == 1
    assert len(recorded.attempts) == 3
    assert recorded.serving_reports == []
    failure = load_record(
        VerificationReport, cohort_ports.store.retrieve(recorded.failed_boots[0]) or {}
    )
    assert failure.failures == ("harness:exception:RuntimeError",)
    assert _cost_on_records(cohort_ports.store) == pytest.approx(
        cohort_ports.budget.spent, abs=1e-4
    )


def test_provision_error_is_settled_and_recorded(tmp_path: Path) -> None:
    cohort_ports, _, targets, _ = ports(tmp_path, _healthy)
    sp = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")

    original = cohort_ports.target_factory

    def failing_factory(requested: Any) -> Any:
        target = original(requested)

        def boom(env: Any = None) -> Any:
            raise ConnectionError("RunPod API unreachable")

        target.provision = boom  # type: ignore[method-assign]
        return target

    object.__setattr__(cohort_ports, "target_factory", failing_factory)
    result = run_cohort([sp], accepted_inputs(), cohort_ports)
    assert "ConnectionError" in result.skipped[sp.label]
    assert targets[0].torn_down
    assert cohort_ports.budget.holds == {}
    assert _cost_on_records(cohort_ports.store) == pytest.approx(
        cohort_ports.budget.spent, abs=1e-4
    )


def test_repeated_unexpected_failures_stop_the_run(tmp_path: Path) -> None:
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy)
    engine.verify = lambda *a, **k: (_ for _ in ()).throw(OSError("ssh dropped"))  # type: ignore[method-assign]
    plans = [
        solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090"),
        solution("Qwen/Qwen3-1.7B", "NVIDIA L4"),
        solution("Qwen/Qwen3-8B", "NVIDIA RTX A6000"),
    ]
    result = run_cohort(plans, accepted_inputs(), cohort_ports)
    assert result.stopped and "2 unexpected failures in a row" in result.stopped
    assert len(targets) == 2
    assert cohort_ports.budget.holds == {}


def test_pod_that_cannot_be_terminated_stops_the_run(tmp_path: Path) -> None:
    """Owner stop condition: the leak is recorded, the pod's still-billing cost
    stays as an open-ended flagged hold, and no further pod is provisioned."""
    from apron.application.orchestration.budget import BudgetTracker
    from apron.application.orchestration.errors import PodLeakError

    cohort_ports, _, targets, events = ports(tmp_path, _healthy)
    original = cohort_ports.target_factory

    def leaking_factory(requested: Any) -> Any:
        target = original(requested)

        def leak() -> None:
            target.torn_down = False
            raise PodLeakError(target.pod_id or "?", "terminate_pod failed 3 times")

        target.teardown = leak  # type: ignore[method-assign]
        return target

    object.__setattr__(cohort_ports, "target_factory", leaking_factory)
    first = solution("Qwen/Qwen3-1.7B", "NVIDIA GeForce RTX 4090")
    second = solution("Qwen/Qwen3-1.7B", "NVIDIA L4")
    result = run_cohort([first, second], accepted_inputs(), cohort_ports)

    assert result.stopped and "PodLeakError" in result.stopped
    assert len(targets) == 1  # the second candidate was never provisioned
    leaked_pod = targets[0].pod_id
    # measurements from the healthy boot are kept; the leak is on a record
    recorded = recorded_evidence(cohort_ports.store, first.solution_fp)
    assert len(recorded.memory_reports) == 1
    leak = [
        load_record(VerificationReport, cohort_ports.store.retrieve(d) or {})
        for d in recorded.failed_boots
    ]
    assert [r.failures for r in leak] == [("harness:pod_not_terminated",)]
    # the settled part equals the cost on records; the leaked pod stays open-ended
    assert _cost_on_records(cohort_ports.store) == pytest.approx(
        cohort_ports.budget.spent, abs=1e-4
    )
    assert f"leak:{leaked_pod}" in cohort_ports.budget.holds
    assert any("open_ended_pod_leak" in f for f in cohort_ports.budget.flags)
    assert any(e["event"] == "pod_leak" for e in events.entries)
    # the next start settles it at RunPod's reported cost
    replayed = BudgetTracker.replay(
        authorized=100.0,
        ledger=cohort_ports.budget.ledger,
        clock=cohort_ports.clock,
        pod_cost=lambda pod: 0.9 if pod == leaked_pod else None,
    )
    assert replayed.holds == {}
    assert replayed.spent == pytest.approx(cohort_ports.budget.spent + 0.9)
