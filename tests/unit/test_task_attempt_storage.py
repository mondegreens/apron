"""F4: task attempts are stored; solution identity is the solution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apron.adapters.backends.local_store import LocalRecordStore
from apron.application.orchestration.evidence import (
    EvidenceContext,
    build_task_attempt,
    solution_fingerprint,
)
from apron.domain.fingerprints import fingerprint_hex
from apron.domain.mechanisms import ComponentMechanism
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.migrations import load_record
from apron.domain.schemas.models import ModelSpec
from apron.domain.schemas.records import TaskAttemptRecord
from apron.domain.schemas.solutions import (
    DeploymentPlan,
    EvaluationProtocol,
    RequestedExecutionSpec,
)
from apron.domain.schemas.tasks import ApplicationSpec, TaskSuiteSpec

RUN = Path(__file__).parents[1] / "fixtures" / "phase-1a-run"
_PLACEHOLDER = "1220" + "00" * 32


def _model(name: str = "Qwen/Qwen3-8B") -> ModelSpec:
    return ModelSpec(
        repository=name,
        components=(ComponentMechanism(mechanism="autoregressive_decode", role="decoder"),),
    )


def _requested(gpu: str = "NVIDIA GeForce RTX 4090", count: int = 1) -> RequestedExecutionSpec:
    return RequestedExecutionSpec(
        provider="runpod",
        gpu_sku=gpu,
        gpu_count=count,
        cloud_type="SECURE",
        image_digest="sha256:" + "a" * 64,
    )


_PLAN = DeploymentPlan(dtype="bfloat16", engine_configuration={"max_model_len": "640"})


def _ctx(solution_fp: str) -> EvidenceContext:
    template = EvaluationProtocol.model_validate_json(
        (RUN / "evaluation-protocol.json").read_text()
    ).model_dump(mode="json")
    for key in (
        "decision_request_digest",
        "task_suite_fingerprint",
        "application_fingerprint",
        "solution_fingerprint",
    ):
        template.pop(key)
    return EvidenceContext.bind(
        request=DecisionRequest.model_validate_json((RUN / "decision-request.json").read_text()),
        task_suite=TaskSuiteSpec.model_validate_json((RUN / "task-suite-spec.json").read_text()),
        application=ApplicationSpec.model_validate_json(
            (RUN / "application-spec.json").read_text()
        ),
        protocol_template=template,
        solution_fp=solution_fp,
    )


# --- solution identity -------------------------------------------------------


def test_same_inputs_same_solution_fingerprint() -> None:
    assert solution_fingerprint(_model(), _PLAN, _requested()) == solution_fingerprint(
        _model(), _PLAN, _requested()
    )


@pytest.mark.parametrize(
    "variant",
    [
        {"model": _model("mistralai/Mistral-7B-Instruct-v0.3")},
        {"plan": _PLAN.model_copy(update={"engine_configuration": {"max_model_len": "4096"}})},
        {"requested": _requested(gpu="NVIDIA L4")},
        {"requested": _requested(count=4)},
    ],
    ids=["model", "plan", "gpu", "gpu_count"],
)
def test_solution_fingerprint_differs_by_model_plan_gpu(variant: dict[str, Any]) -> None:
    base = solution_fingerprint(_model(), _PLAN, _requested())
    other = solution_fingerprint(
        variant.get("model", _model()),
        variant.get("plan", _PLAN),
        variant.get("requested", _requested()),
    )
    assert base != other


def test_solution_fingerprint_ignores_plan_display_fields() -> None:
    rendered = _PLAN.model_copy(update={"serve_command": "vllm serve x"})
    assert solution_fingerprint(_model(), _PLAN, _requested()) == solution_fingerprint(
        _model(), rendered, _requested()
    )


def test_solution_fingerprint_is_not_the_model_fingerprint() -> None:
    assert solution_fingerprint(_model(), _PLAN, _requested()) != fingerprint_hex(_model())


# --- attempt records ---------------------------------------------------------


def test_passed_and_failed_cases_are_first_class_attempts() -> None:
    sol = solution_fingerprint(_model(), _PLAN, _requested())
    ctx = _ctx(sol)
    passed = build_task_attempt(
        ctx,
        {"case_id": "arith-1", "output": "4", "score": 1, "accepted": True, "status": "completed"},
        attempt_id="a1",
        infrastructure_cost=0.002,
    )
    failed = build_task_attempt(
        ctx,
        {"case_id": "fact-1", "score": 0, "accepted": False, "status": "failed", "error": "503"},
        attempt_id="a2",
        retry=1,
        infrastructure_cost=0.002,
    )
    assert passed.accepted is True and passed.failures == ()
    assert failed.accepted is False
    assert failed.failures == ("evaluation:503",)
    assert failed.retries == 1
    for record in (passed, failed):
        assert record.solution_fingerprint == sol
        assert record.decision_fingerprint == fingerprint_hex(ctx.request)
        assert record.evaluation_protocol_fingerprint == fingerprint_hex(ctx.protocol)
        assert record.infrastructure_cost == 0.002
        assert _PLACEHOLDER not in record.model_dump_json()


def test_protocol_template_cannot_preset_digests() -> None:
    from apron.application.orchestration.evidence import build_evaluation_protocol

    ctx = _ctx(solution_fingerprint(_model(), _PLAN, _requested()))
    with pytest.raises(ValueError, match="bound fields"):
        build_evaluation_protocol(
            {"harness": "h", "harness_version": "1", "scorer": "s", "solution_fingerprint": "x"},
            request=ctx.request,
            task_suite=ctx.task_suite,
            application=ctx.application,
            solution_fp="1220" + "ab" * 32,
        )


# --- the CLI stores one record per case ---------------------------------------


def test_cli_run_task_suite_stores_one_record_per_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apron.adapters.evaluations import deterministic_scorer
    from apron.interfaces import cli

    def fake_execute(self, protocol, endpoint):  # type: ignore[no-untyped-def]
        return [
            {"case_id": c["id"], "output": c["expected"], "score": 1, "accepted": True}
            if c["id"] != "fact-1"
            else {
                "case_id": c["id"],
                "score": 0,
                "accepted": False,
                "status": "failed",
                "error": "x",
            }
            for c in protocol["cases"]
        ]

    monkeypatch.setattr(deterministic_scorer.DeterministicScorer, "execute", fake_execute)

    class _Target:
        proxy_url = "https://pod-8000.proxy.runpod.net"

    store = LocalRecordStore(tmp_path)
    sol = solution_fingerprint(_model(), _PLAN, _requested())
    cli._run_task_suite(
        RUN / "task-suite-spec.json",
        request=DecisionRequest.model_validate_json((RUN / "decision-request.json").read_text()),
        model_id="Qwen/Qwen3-8B",
        solution_fp=sol,
        target=_Target(),
        store=store,
        report_digest="1220" + "cd" * 32,
    )
    stored = [
        load_record(TaskAttemptRecord, store.retrieve(d) or {}) for d in store.search("1220")
    ]
    assert sorted(r.case_id for r in stored) == ["arith-1", "arith-2", "fact-1"]
    assert {r.solution_fingerprint for r in stored} == {sol}
    assert [r.accepted for r in sorted(stored, key=lambda r: r.case_id)] == [True, True, False]
    assert all(r.trace_references == ("1220" + "cd" * 32,) for r in stored)
