"""Pod reuse (TargetPool): one pod per requested execution for a run.

L0 spent most pod time pulling the runner image for boots of seconds; the pool
keeps a pod for the next solution on the same execution.  These tests use the
real orchestrator, budget ledger and records with the mock provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from _cohort_fakes import MemoryLog, accepted_inputs, ports, solution

from apron.application.orchestration.budget import BudgetTracker
from apron.application.orchestration.cohort import run_cohort

if TYPE_CHECKING:
    from pathlib import Path

_4090, _L4 = "NVIDIA GeForce RTX 4090", "NVIDIA L4"


def _healthy(plan: Any, target: Any) -> str:
    return "healthy"


def _cost_on_records(store: Any) -> float:
    total = 0.0
    for digest in store.search(""):
        raw = store.retrieve(digest) or {}
        if raw.get("claim_scope") == "task_outcome":
            total += raw.get("infrastructure_cost") or 0.0
        elif raw.get("claim_scope") in ("boot", "memory", "serving_performance"):
            total += raw.get("market_equivalent_price") or 0.0
    return total


def _ledger_by_prefix(ledger: MemoryLog, prefix: str) -> float:
    return sum(
        float(e["amount"])
        for e in ledger.read_all()
        if e["op"] == "settle" and str(e["label"]).startswith(prefix)
    )


def test_same_execution_shares_one_pod(tmp_path: Path) -> None:
    ledger = MemoryLog()
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy, ledger=ledger, pool=True)
    plans = [
        solution("Qwen/Qwen3-1.7B", _4090),
        solution("mistralai/Mistral-7B-Instruct-v0.3", _4090),
    ]
    result = run_cohort(plans, accepted_inputs(), cohort_ports)
    assert result.stopped is None
    assert len(targets) == 1, "the second solution must reuse the pod"
    assert targets[0].torn_down, "the pool tears the pod down when the run ends"
    assert engine.evicted == ["mistralai/Mistral-7B-Instruct-v0.3"]
    assert all(o.healthy for o in result.executed.values())
    # every paid second is on a record or in the ledger's idle line; nothing is open
    assert cohort_ports.budget.holds == {}
    solutions = sum(
        float(e["amount"])
        for e in ledger.read_all()
        if e["op"] == "settle" and not str(e["label"]).startswith("pod-idle:")
    )
    assert _cost_on_records(cohort_ports.store) == pytest.approx(solutions, abs=1e-4)
    assert _ledger_by_prefix(ledger, "pod-idle:") > 0


def test_different_executions_get_different_pods(tmp_path: Path) -> None:
    cohort_ports, _, targets, _ = ports(tmp_path, _healthy, pool=True)
    plans = [solution("Qwen/Qwen3-1.7B", _4090), solution("Qwen/Qwen3-1.7B", _L4)]
    run_cohort(plans, accepted_inputs(), cohort_ports)
    assert [t.requested.gpu_sku for t in targets] == [_4090, _L4]
    assert all(t.torn_down for t in targets)


def test_a_pod_is_not_reused_after_a_harness_failure(tmp_path: Path) -> None:
    cohort_ports, engine, targets, _ = ports(tmp_path, _healthy, pool=True)
    calls = {"n": 0}

    def dirty_once(target: Any) -> dict[str, Any]:
        calls["n"] += 1
        return {"clean": calls["n"] > 2}  # both attempts of the first solution unclean

    engine.prepare_boot = dirty_once  # type: ignore[method-assign]
    plans = [
        solution("Qwen/Qwen3-1.7B", _4090),
        solution("mistralai/Mistral-7B-Instruct-v0.3", _4090),
    ]
    run_cohort(plans, accepted_inputs(), cohort_ports)
    assert len(targets) == 2, "an unclean GPU's pod must be dropped, not reused"
    assert targets[0].torn_down and targets[1].torn_down


def test_a_parked_pod_holds_budget_so_a_crash_is_accounted(tmp_path: Path) -> None:
    ledger = MemoryLog()
    cohort_ports, _, targets, _ = ports(tmp_path, _healthy, ledger=ledger, pool=True)
    from apron.application.orchestration.cohort import execute_solution

    execute_solution(solution("Qwen/Qwen3-1.7B", _4090), accepted_inputs(), cohort_ports)
    # the process "dies" here: the pod is parked, its idle hold open and annotated
    open_holds = [
        e for e in ledger.read_all() if e["op"] == "hold" and e["label"].startswith("pod-idle:")
    ]
    assert len(open_holds) == 1
    replayed = BudgetTracker.replay(
        authorized=100.0,
        ledger=ledger,
        clock=cohort_ports.clock,
        pod_cost=lambda pod: 0.5 if pod == targets[0].pod_id else None,
    )
    assert replayed.holds == {}
    assert any("pod-idle:" in f and "replayed:provider_reported" in f for f in replayed.flags)


def _with_capacity(cohort_ports: Any, answer: Any) -> list[str]:
    asked: list[str] = []

    def await_capacity(requested: Any) -> bool:
        asked.append(requested.gpu_sku)
        return answer(requested)

    object.__setattr__(cohort_ports, "await_capacity", await_capacity)
    return asked


def test_no_stock_skips_without_a_pod_or_a_circuit_break(tmp_path: Path) -> None:
    cohort_ports, _, targets, _ = ports(tmp_path, _healthy, pool=True)
    _with_capacity(cohort_ports, lambda r: r.gpu_sku != _L4)
    plans = [
        solution("Qwen/Qwen3-1.7B", _L4),
        solution("mistralai/Mistral-7B-Instruct-v0.3", _L4, label="second-l4"),
        solution("Qwen/Qwen3-1.7B", _4090),
    ]
    result = run_cohort(plans, accepted_inputs(), cohort_ports)
    assert result.stopped is None, "capacity is not an unexpected failure"
    assert result.skipped[plans[0].label] == "no provider capacity within the wait"
    assert [t.requested.gpu_sku for t in targets] == [_4090]
    assert plans[2].label in result.executed


def test_a_refused_creation_waits_and_retries(tmp_path: Path) -> None:
    cohort_ports, _, targets, _ = ports(tmp_path, _healthy, pool=True)
    _with_capacity(cohort_ports, lambda r: True)
    original = cohort_ports.target_factory
    refusals = {"left": 2}

    def refusing(requested: Any) -> Any:
        target = original(requested)
        real = target.provision

        def provision(env: Any = None, wait_timeout: int = 0) -> Any:
            if refusals["left"]:
                refusals["left"] -= 1
                raise RuntimeError(
                    "There are no longer any instances available with the requested "
                    "specifications. Please refresh and try again."
                )
            return real(env=env, wait_timeout=wait_timeout)

        target.provision = provision  # type: ignore[method-assign]
        return target

    object.__setattr__(cohort_ports.pool, "factory", refusing)
    sp = solution("Qwen/Qwen3-1.7B", _4090)
    result = run_cohort([sp], accepted_inputs(), cohort_ports)
    assert sp.label in result.executed and sp.label not in result.skipped
    assert result.stopped is None
    assert len(targets) == 3  # two refused creations, one pod
    assert cohort_ports.budget.holds == {}


def test_a_reused_pod_needs_no_stock_wait(tmp_path: Path) -> None:
    cohort_ports, _, targets, _ = ports(tmp_path, _healthy, pool=True)
    asked = _with_capacity(cohort_ports, lambda r: True)
    plans = [
        solution("Qwen/Qwen3-1.7B", _4090),
        solution("mistralai/Mistral-7B-Instruct-v0.3", _4090),
    ]
    run_cohort(plans, accepted_inputs(), cohort_ports)
    assert asked == [_4090]  # only the first solution needed a new pod
    assert len(targets) == 1
