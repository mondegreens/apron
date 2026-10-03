"""scorer_input: every protocol field is sent, or the protocol is refused.

L5 review (2026-09-27): ``sampling_top_p``, ``stopping_rules``,
``repetitions`` and ``concurrency`` were fingerprinted but never reached the
harness — a protocol asking for five repetitions ran once, silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apron.application.orchestration.evidence import EvidenceContext, scorer_input
from apron.domain.schemas.authority import DecisionRequest
from apron.domain.schemas.solutions import EvaluationProtocol
from apron.domain.schemas.tasks import ApplicationSpec, TaskSuiteSpec

RUN = Path(__file__).resolve().parents[1] / "fixtures" / "phase-1a-run"


def _ctx(**protocol: object) -> EvidenceContext:
    template = json.loads((RUN / "evaluation-protocol.json").read_text())
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
        protocol_template={**template, **protocol},
        solution_fp="1220" + "ab" * 32,
    )


def test_sampling_fields_reach_the_scorer() -> None:
    data = scorer_input(_ctx(sampling_top_p=0.9, stopping_rules=["\n"]), model_id="m")
    assert data["sampling_top_p"] == 0.9
    assert data["stopping_rules"] == ["\n"]


def test_unset_sampling_fields_are_absent() -> None:
    data = scorer_input(_ctx(), model_id="m")
    assert "sampling_top_p" not in data and "stopping_rules" not in data


@pytest.mark.parametrize(("field", "value"), [("repetitions", 3), ("concurrency", 4)])
def test_what_the_harness_cannot_do_is_refused(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        scorer_input(_ctx(**{field: value}), model_id="m")


def test_protocol_type_is_the_schema() -> None:
    assert isinstance(_ctx().protocol, EvaluationProtocol)


def test_a_cut_answer_is_marked_on_its_record() -> None:
    from apron.application.orchestration.evidence import build_task_attempt

    ctx = _ctx()
    scored = {
        "case_id": "arith-2",
        "output": "10 * 5 = 5",
        "score": 0,
        "accepted": False,
        "status": "completed",
        "finish_reason": "length",
    }
    record = build_task_attempt(ctx, scored, attempt_id="a1")
    assert "evaluation:truncated at max_tokens" in record.failures
