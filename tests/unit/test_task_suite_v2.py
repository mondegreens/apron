"""Task suite v2 for the modern models: same questions, room to reason first.

Owner, 2026-09-27: 8 answer tokens are spent before a reasoning model
answers (gpt-oss always reasons; MiniMax-M2.7 always thinks).  The longest
rendered prompt of the group A/B models is gpt-oss's, 81 tokens (their chat
templates, rendered locally): 81 + 512 fits the plans' 640-token context.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from apron.adapters.evaluations.deterministic_scorer import DeterministicScorer
from apron.application.orchestration.evidence import (
    chat_template_kwargs,
    reasoning_request_fields,
)
from apron.domain.schemas.tasks import TaskSuiteSpec
from apron.interfaces.cohort_root import FIXTURES, TASK_SUITE_V2, load_inputs


def test_v2_asks_the_same_questions_with_512_answer_tokens() -> None:
    v1 = json.loads((FIXTURES / "task-suite-spec.json").read_text())
    v2 = TaskSuiteSpec.model_validate_json(TASK_SUITE_V2.read_text())
    assert [(c["id"], c["prompt"], c["expected"]) for c in v2.cases] == [
        (c["id"], c["prompt"], c["expected"]) for c in v1["cases"]
    ]
    assert {c["max_tokens"] for c in v2.cases} == {"512"}
    assert load_inputs(task_suite=TASK_SUITE_V2).task_suite == v2
    assert load_inputs().task_suite.cases[0]["max_tokens"] == "8"  # the first cohort's


def test_reasoning_is_switched_off_where_it_can_be_and_kept_low_where_not() -> None:
    qwen = "{% if enable_thinking %}...{% endif %} reasoning_effort"
    gpt_oss = "{% set reasoning_effort = reasoning_effort | default('medium') %}"
    plain = "{{ messages }}"
    assert chat_template_kwargs(qwen) == {"enable_thinking": False}
    assert reasoning_request_fields(qwen) is None  # the switch wins
    assert chat_template_kwargs(gpt_oss) is None
    assert reasoning_request_fields(gpt_oss) == {"reasoning_effort": "low"}
    assert reasoning_request_fields(plain) is None


def test_the_scorer_sends_the_reasoning_field_in_the_request_body() -> None:
    sent: list[dict[str, Any]] = []

    class _Response:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]}

    def post(url: str, json: dict[str, Any], timeout: float) -> _Response:
        sent.append(json)
        return _Response()

    scorer = DeterministicScorer()
    prepared = scorer.prepare(
        {
            "cases": [{"id": "arith-1", "prompt": "2 + 2?", "expected": "4", "max_tokens": "512"}],
            "model_id": "m",
            "endpoint": "http://x",
            "deterministic_checks": ["whitespace_normalized_exact_match"],
            "request_fields": {"reasoning_effort": "low"},
        }
    )
    with patch("apron.adapters.evaluations.deterministic_scorer.httpx.post", post):
        scorer.execute(prepared, "http://x")
    assert sent[0]["reasoning_effort"] == "low"
    assert sent[0]["max_tokens"] == 512
